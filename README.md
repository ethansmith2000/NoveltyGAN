# NoveltyGAN


A GAN objective combined with a **negative** density loss evaluated by a seperate pretrained frozen density model. The GAN objective asks that generated images be perceptually convincing, but then the trick is we use a pretrained density model (Diffusion, autoregressive, normalizing flow, VAE ...) and flip the sign of the loss, which asks generated samples to be lower probability, prompting exploration of rarer regions of space. In this implementation GAN generator is initialized from a pretrained diffusion model and the same frozen model is used as the density model, something like normalizing flow, which has recently showed the capability to make for expressive generative models on diverse datasets, feels like it may fit better here. One possible issue is that instead of promoting exploration, we could still have mode collapse but just in a less dense region of space, so still low diversity, still a narrow mode that can trick the discriminator, but less probable outputs at some pareto frontier. 

To avoid this there may be some strategy of caching past N generations and comparing them to the generators output, increasing penalty more and more as generations collapse into sameness if they do. Also it may make sense to optimize the density model online as well, though while still using a stop-grad for its density estimate, just to make sure the domain is aligned between the two.
