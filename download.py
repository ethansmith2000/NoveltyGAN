import asyncio
import aiohttp
import pandas as pd
import os
from tqdm.asyncio import tqdm

def read_df(path):
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.csv'):
        return pd.read_csv(path)
    else:
        raise ValueError("File format not supported. Please provide a .parquet or .csv file.")

async def download_image(session, url, sha256, folder, pbar, semaphore):
    try:
        async with semaphore:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    file_name = f"{sha256}.jpg"
                    file_path = os.path.join(folder, file_name)
                    with open(file_path, 'wb') as f:
                        f.write(await response.read())
                    pbar.update(1)
                    return True
                else:
                    print(f"Failed to download: {url}")
                    pbar.update(1)
                    return False
    except asyncio.TimeoutError:
        print(f"Timeout downloading: {url}")
        pbar.update(1)
        return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        pbar.update(1)
        return False

async def download_all_images(df, url_column, sha256_column, folder):
    os.makedirs(folder, exist_ok=True)
    
    # Limit concurrent connections
    semaphore = asyncio.Semaphore(10)
    
    async with aiohttp.ClientSession() as session:
        total_images = len(df)
        tasks = []
        
        with tqdm(total=total_images, desc="Downloading Images") as pbar:
            for url, sha256 in zip(df[url_column], df[sha256_column]):
                task = asyncio.ensure_future(download_image(session, url, sha256, folder, pbar, semaphore))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_downloads = sum(1 for r in results if r is True)
        print(f"Successfully downloaded {successful_downloads} out of {total_images} images.")

# Example usage
df = read_df('/home/ubuntu/NoveltyGAN/data/combined.parquet') 
url_column = 'url'
sha256_column = 'sha256'
download_folder = 'images'

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# Run the asynchronous download
asyncio.run(download_all_images(df, url_column, sha256_column, download_folder))