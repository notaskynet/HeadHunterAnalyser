import pandas as pd
import requests
import aiohttp
import asyncio
import random
from typing import List, Dict

NUMBER_OF_PAGES = 100

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.80 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/81.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 13_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Mobile/15E148 Safari/604.1',
]

async def fetch_page(session, url, params, semaphore):
    for attempt in range(3):
        async with semaphore:
            headers = {'User-Agent': random.choice(user_agents)}
            
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    await asyncio.sleep(2 ** attempt)
                elif response.status == 403 or response.status == 400:
                    return None
                else:
                    response.raise_for_status()
        await asyncio.sleep(random.uniform(0.5, 1.5))

async def parse_job(job_title: str = "", number_of_pages: int = NUMBER_OF_PAGES) -> List:
    data = []
    url = 'https://api.hh.ru/vacancies'
    semaphore = asyncio.Semaphore(3)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for page in range(number_of_pages):
            params = {'text': job_title, 'area': '113', 'per_page': '10', 'page': page}
            tasks.append(fetch_page(session, url, params, semaphore))

        responses = await asyncio.gather(*tasks)

        for resp in responses:
            if resp is None:
                continue
            elif resp and "items" in resp and len(resp['items']) > 0:
                data.extend(resp['items'])
    return data

def get_key_words(keywords: List[str] = []) -> str:
    keywords = [f"'{x}'" for x in keywords]
    lower_keywords = [f"{x.lower()}" for x in keywords]
    return f'{" OR ".join(keywords)} OR {" OR ".join(set(keywords) - set(lower_keywords))}'
