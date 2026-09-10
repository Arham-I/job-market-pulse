import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def get_DBconnection(): #Connecting to database 
    return psycopg2.connect(
        dbname="job_market",
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        # host="localhost",
        # port="5432"
    )

def insert_jobs(jobs):
    conn = get_DBconnection()
    cursr = conn.cursor()
    inserted = 0
    skipped = 0

    for job in jobs:
        try:
            cursr.execute("""
                INSERT INTO jobs (
                    adzuna_id, title, search_term, company,
                    location, category, salary_min, salary_max,
                    salary_is_predicted, contract_time, contract_type,
                    description, created)
                VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (adzuna_id) DO NOTHING"""
            ,(
                job.get("id"),
                job.get("title"),
                job.get("search_term"),
                job.get("company", {}).get("display_name"),
                # job.get("company", {}).get("average_salary"),
                job.get("location", {}).get("display_name"),
                job.get("category", {}).get("label"),
                job.get("salary_min"),
                job.get("salary_max"),
                job.get("salary_is_predicted") == "1",
                job.get("contract_time"),
                job.get("contract_type"),
                job.get("description"),
                datetime.fromisoformat(job.get("created", "").replace("Z", "+00:00")) if job.get("created") else None
            ))
            inserted += 1
        except Exception as e:
            print(f"Error inserting job {job.get('id')} into DB: {e}")
            skipped += 1

    conn.commit()
    cursr.close()
    conn.close()

    print(f"Inserted: {inserted} | Skipped: {skipped}")

if __name__ == "__main__":
    from fetch import fetch_all_jobs
    jobs = fetch_all_jobs()
    insert_jobs(jobs)
