import pytest
import asyncio
import aiohttp
from unittest.mock import patch
from aioresponses import aioresponses
from utils.parsing import fetch_page, parse_job, get_key_words

@pytest.mark.asyncio
async def test_fetch_page_success():
    url = "https://api.hh.ru/vacancies"
    params = {"text": "developer", "area": "113", "per_page": "10", "page": 1}
    
    with aioresponses() as m:
        m.get(url, payload={"items": [{"id": 1, "name": "Job 1"}, {"id": 2, "name": "Job 2"}]})
        
        async with aiohttp.ClientSession() as session:
            result = await fetch_page(session, url, params, asyncio.Semaphore(1))
    
    assert result == {"items": [{"id": 1, "name": "Job 1"}, {"id": 2, "name": "Job 2"}]}

@pytest.mark.asyncio
async def test_fetch_page_rate_limit():
    url = "https://api.hh.ru/vacancies"
    params = {"text": "developer", "area": "113", "per_page": "10", "page": 1}
    
    with aioresponses() as m:
        m.get(url, status=429)
        
        async with aiohttp.ClientSession() as session:
            result = await fetch_page(session, url, params, asyncio.Semaphore(1))
    
    assert result is None

@pytest.mark.asyncio
async def test_fetch_page_forbidden():
    url = "https://api.hh.ru/vacancies"
    params = {"text": "developer", "area": "113", "per_page": "10", "page": 1}
    
    with aioresponses() as m:
        m.get(url, status=403)
        
        async with aiohttp.ClientSession() as session:
            result = await fetch_page(session, url, params, asyncio.Semaphore(1))
    

    assert result is None


@pytest.mark.asyncio
async def test_parse_job():
    job_title = "developer"
    number_of_pages = 1
    url = "https://api.hh.ru/vacancies"

    with aioresponses() as m:
        m.get(url, payload={"items": [{"id": 1, "name": "Job 1"}]})
        
        result = await parse_job(job_title, number_of_pages)
    
    assert len(result) == 1
    assert result[0]["name"] == "Job 1"


def test_get_key_words():
    keywords = ["developer", "engineer"]
    
    result = get_key_words(keywords)
    
    assert result == "'developer' OR 'engineer' OR 'developer' OR 'engineer'"

    result_empty = get_key_words()
    assert result_empty == ""