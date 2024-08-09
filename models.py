import torch
import torch.nn as nn
import torch.nn.functional as F
import abc

from types import MethodType

from typing import Any, Dict, Optional, Tuple, Union


def weight_init(weight, act="relu", weight_init_method="xavier", mode="uniform", gain=1.0):
    mapping = {
        "kaiming_uniform": nn.init.kaiming_uniform_,
        "kaiming_normal": nn.init.kaiming_normal_,
        "xavier_uniform": nn.init.xavier_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
    }
    kwargs = {"nonlinearity": act} if "kaiming" in weight_init_method else {"gain": gain}
    weight_init_method = weight_init_method + "_" + mode
    mapping[weight_init_method](weight, **kwargs)


def make_linear(in_dim, 
                out_dim, 
                bias=True, 
                act="relu", 
                weight_init_method="xavier"
                ):
    linear = nn.Linear(in_dim, out_dim, bias=bias)
    weight_init(linear.weight, act=act, weight_init_method=weight_init_method)
    if bias:
        nn.init.zeros_(linear.bias)
    return linear


def unet_encoder_forward(
    self,
    sample: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):

    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None

    class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None:
        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    aug_emb = self.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )
    if self.config.addition_embed_type == "image_hint":
        aug_emb, hint = aug_emb
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    # 2.5 GLIGEN position net
    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

    # 3. down
    # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
    # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
    if cross_attention_kwargs is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        lora_scale = cross_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0


    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
    # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
    is_adapter = down_intrablock_additional_residuals is not None
    # maintain backward compatibility for legacy usage, where
    #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
    #       but can only use one or the other
    if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
        down_intrablock_additional_residuals = down_block_additional_residuals
        is_adapter = True

    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                sample += down_intrablock_additional_residuals.pop(0)

        down_block_res_samples += res_samples

    if is_controlnet:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals
        ):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = self.mid_block(sample, emb)

        # To support T2I-Adapter-XL
        if (
            is_adapter
            and len(down_intrablock_additional_residuals) > 0
            and sample.shape == down_intrablock_additional_residuals[0].shape
        ):
            sample += down_intrablock_additional_residuals.pop(0)

    if is_controlnet:
        sample = sample + mid_block_additional_residual

    return sample, down_block_res_samples



class DensityEstimator(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, x, extra_kwargs):
        pass


class UNetDensityEstimator(DensityEstimator):
    def __init__(self, unet, noise_scheduler):
        super().__init__()
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.min_timestep = 0
        self.max_timestep = 1000


    def forward(self, x, extra_kwargs):
        # Sample forward diffusion process timestep
        timesteps = torch.randint(
            self.min_timestep,
            self.max_timestep,
            (x.shape[0],),
            device=x.device,
        ).long()

        # Sample noise to predict
        noise = torch.randn_like(x)
        xt = self.noise_scheduler.add_noise(x, noise, timesteps).to(x.dtype)


        noise_pred = self.unet(xt,
                                timesteps,
                                # encoder_hidden_states,
                                # added_cond_kwargs=added_cond_kwargs
                                return_dict=False,
                                **extra_kwargs
                                )[0]

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss * -1.0


class VAEDensityEstimator(DensityEstimator):

    def __init__(self, vae):
        super().__init__()
        self.vae = vae



class Attention(nn.Module):
    def __init__(self, heads, dim, dropout=0.0):
        super().__init__()
        self.h = heads
        self.d = dim
        self.dropout = nn.Dropout(dropout)
        self.qkv = make_linear(dim, dim * 3, bias=False)
        self.out = make_linear(dim, dim)

    def forward(self, x):
        q, k, v = map(lambda t: t.reshape(*t.shape[:-1], self.h, self.d // self.h).transpose(1, 2), self.qkv(x).chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v).reshape(*x.shape)
        return self.out(out)


class CrossAttention(nn.Module):
    def __init__(self, heads, dim, dropout=0.0):
        super().__init__()
        self.h = heads
        self.d = dim
        self.dropout = nn.Dropout(dropout)
        self.q = make_linear(dim, dim, bias=False)
        self.kv = make_linear(dim, dim * 2, bias=False)
        self.out = make_linear(dim, dim)

    def forward(self, x, context):
        k, v = map(lambda t: t.reshape(*t.shape[:-1], self.h, self.d // self.h).transpose(1, 2), (self.to_q(x), *self.kv(context).chunk(2, dim=-1)))
        out = F.scaled_dot_product_attention(q, k, v).reshape(*x.shape)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            make_linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            make_linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, 
                        heads, 
                        mlp_mult, 
                        dropout=0.0,
                        cross_attn_dim=None,
                        ada_norm_dim=None,
                        ):
        super().__init__()
        self.attn = Attention(heads, dim, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.cross_attn = None
        if cross_attn_dim is not None:
            self.cross_attn = CrossAttention(heads, cross_attn_dim, dropout)
        
        if ada_norm_dim is not None:
            self.norm_attn = nn.LayerNorm(ada_norm_dim)
            self.norm_ff = nn.LayerNorm(ada_norm_dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        pass
        


#TODO this needs more
class Discriminator(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.norm_out = nn.GroupNorm(32, 1280)
        self.fc_out = make_linear(1280, 1)
        self.unet.up_blocks = None # remove up_blocks
        self.unet.forward = MethodType(unet_encoder_forward, self.unet)

    def forward(self, x, t, extra_kwargs):
        pred = self.unet(x, t, return_dict=False, **extra_kwargs)[0]
        pred = self.norm_out(pred)
        pred = self.fc_out(pred.permute(0, 2, 3, 1)).squeeze(-1)
        return pred
        
