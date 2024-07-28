import asyncio
import aiohttp
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

def read_df(path):
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.csv'):
        return pd.read_csv(path)
    else:
        raise ValueError("File format not supported. Please provide a .parquet or .csv file.")



async def download_image(session, url, sha256, folder):
    async with session.get(url) as response:
        if response.status == 200:
            file_name = f"{sha256}.jpg"
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'wb') as f:
                f.write(await response.read())
            return True
        else:
            print(f"Failed to download: {url}")
            return False

async def download_chunk(chunk, folder):
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, row['url'], row['sha256'], folder) for _, row in chunk.iterrows()]
        results = await atqdm.gather(*tasks, desc="Downloading")
    return sum(results)

def process_chunk(chunk, folder):
    return asyncio.run(download_chunk(chunk, folder))

def parallel_download(df, url_column, sha256_column, folder, num_processes=4):
    os.makedirs(folder, exist_ok=True)
    
    # Split the DataFrame into chunks
    chunk_size = len(df) // num_processes
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Create a partial function with fixed arguments
    download_func = partial(process_chunk, folder=folder)
    
    total_images = len(df)
    downloaded_images = 0
    
    # Use ProcessPoolExecutor to parallelize the downloads
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        with tqdm(total=total_images, desc="Total Progress") as pbar:
            for result in executor.map(download_func, chunks):
                downloaded_images += result
                pbar.update(len(chunks[0]))  # Update by chunk size
    
    print(f"Downloaded {downloaded_images} out of {total_images} images.")

# Example usage
df = read_df('/home/ubuntu/NoveltyGAN/data/combined.parquet') 
url_column = 'url'
sha256_column = 'sha256'
download_folder = 'images'

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

parallel_download(df, url_column, sha256_column, download_folder)




# import asyncio
# import aiohttp
# import pandas as pd
# import os
# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
# from tqdm import tqdm
# from tqdm.asyncio import tqdm as atqdm

# async def download_image(session, url, sha256, folder, semaphore):
#     try:
#         async with semaphore:
#             async with session.get(url, timeout=30) as response:
#                 if response.status == 200:
#                     file_name = f"{sha256}.jpg"
#                     file_path = os.path.join(folder, file_name)
#                     with open(file_path, 'wb') as f:
#                         f.write(await response.read())
#                     return True
#                 else:
#                     print(f"Failed to download: {url}")
#                     return False
#     except asyncio.TimeoutError:
#         print(f"Timeout downloading: {url}")
#         return False
#     except Exception as e:
#         print(f"Error downloading {url}: {str(e)}")
#         return False

# async def download_chunk(chunk, folder):
#     semaphore = asyncio.Semaphore(10)  # Limit concurrent downloads
#     async with aiohttp.ClientSession() as session:
#         tasks = [download_image(session, row['url'], row['sha256'], folder, semaphore) 
#                  for _, row in chunk.iterrows()]
#         results = await atqdm.gather(*tasks, desc="Downloading")
#     return sum(results)

# def process_chunk(chunk, folder):
#     return asyncio.run(download_chunk(chunk, folder))

# def parallel_download(df, url_column, sha256_column, folder, num_processes=4):
#     os.makedirs(folder, exist_ok=True)
    
#     # Split the DataFrame into chunks
#     chunk_size = len(df) // num_processes
#     chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
#     # Create a partial function with fixed arguments
#     download_func = partial(process_chunk, folder=folder)
    
#     total_images = len(df)
#     downloaded_images = 0
    
#     # Use ProcessPoolExecutor to parallelize the downloads
#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         with tqdm(total=total_images, desc="Total Progress") as pbar:
#             for result in executor.map(download_func, chunks):
#                 downloaded_images += result
#                 pbar.update(len(chunks[0]))  # Update by chunk size
    
#     print(f"Downloaded {downloaded_images} out of {total_images} images.")

# # Example usage
# df = pd.read_csv('your_dataframe.csv')
# df = df.rename(columns={'image_url': 'url', 'sha256': 'sha256'})
# download_folder = 'downloaded_images'

# parallel_download(df, 'url', 'sha256', download_folder)