import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

BASE_URL = "https://api.adzuna.com/v1/api/jobs/gb/search/"

SEARCH_TERMS = [
    "data scientist",
    "data analyst",
    "data engineer",
    "machine learning engineer",
    "business intelligence analyst"
]

PAGES = 5
RESULTS_PER_PAGE = 50


def fetch_jobs(search_term, page):
    url = f"{BASE_URL}{page}"
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": RESULTS_PER_PAGE,
        "what": search_term
    }
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching '{search_term}' page {page}: {response.status_code}")
        return []
    
    data = response.json()
    return data.get("results", [])


def fetch_all_jobs():
    all_jobs = []

    for term in SEARCH_TERMS:
        print(f"Fetching: {term}")
        for page in range(1, PAGES + 1):
            jobs = fetch_jobs(term, page)
            for job in jobs:
                job["search_term"] = term  # tag which term found this job
            all_jobs.extend(jobs)
            print(f"  Page {page}: {len(jobs)} jobs fetched")
            time.sleep(1)  # polite delay between requests
    
    print(f"\nTotal jobs fetched: {len(all_jobs)}")
    return all_jobs


if __name__ == "__main__":
    jobs = fetch_all_jobs()
    print(jobs[:5])