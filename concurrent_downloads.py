import aiohttp
import asyncio
import os
import argparse
import pandas as pd
from io import BytesIO
from PIL import Image
import math
from tqdm import tqdm
import random

async def download_image(session, url, filename, idx, pbar):
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"Failed to download {url}. Status: {resp.status}","file:",idx)
                pbar.update(1)
                return

            image_data = await resp.read()
            with open(filename, 'wb') as f:
                f.write(image_data)

            pbar.update(1)

    except Exception as e:
        print(f"Error downloading {url}: {e}")


async def main():
    root = "./images"
    df_path = "./dataset.parquet"
    if not os.path.exists(root):
        os.makedirs(root)

    df = pd.read_parquet(df_path)
    urls = df["image_url"].tolist()
    tasks = []
    extension = ".png"

    df["filename"] = None

    try:
        async with aiohttp.ClientSession() as session:
            pbar = tqdm(total=len(urls), desc="Downloading", ncols=100)

            for i, url in enumerate(urls):
                fname = str(random.getrandbits(128)) + extension
                filename = os.path.join(root, fname)
                df.at[i, "filename"] = filename
                tasks.append(download_image(session, url, filename, i, pbar))
                if i >= 20_000:
                    break

            sem = asyncio.Semaphore(50)

            async def bound_download(task):
                async with sem:
                    await task

            await asyncio.gather(*(bound_download(task) for task in tasks))
    except KeyboardInterrupt:
        print("Download interrupted")
        df.to_parquet(df_path, index=False)
        exit(1)
    
    df.to_parquet(df_path, index=False)

    


asyncio.run(main())




