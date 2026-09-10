import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_ID = os.getenv("ADZUNA_APP_ID")
API_KEY = os.getenv("ADZUNA_APP_KEY")

url = "https://api.adzuna.com/v1/api/jobs/gb/search/1"

params = {
    "app_id": API_ID,
    "app_key": API_KEY,
    "results_per_page": 5,
    "what": "data scientist",
}

response = requests.get(url, params)
data = response.json()

count = 0
for job in data["results"]:
    count += 1
    print("Job: ",count) 
    print(job["title"])
    print(job["location"]["display_name"])
    print(job["company"]["display_name"])
    print(job["description"])
    print(job.get("salary_min"), "-", job.get("salary_max"))
    print("-----")
    