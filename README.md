# 🇬🇧 Job Market Pulse — UK DS/ML Job Market Analyser

An end-to-end data science project that fetches real UK data science and machine learning job postings, stores them in a PostgreSQL database, analyses salary trends, and predicts salaries using an XGBoost regression model — all presented via an interactive Streamlit dashboard.

---

## 📊 Live Dashboard
[Link to deployed Streamlit app] ← add after deployment

---

## 🗂️ Project Structure

```
job-market-pulse/
│
├── data/
│   └── raw/                  # raw API responses
├── models/                   # saved model artefacts
│   ├── model.pkl
│   ├── tfidf.pkl
│   ├── selector.pkl
│   ├── ordinal_encoder.pkl
│   └── skill_cols.pkl
├── notebooks/
│   └── 01_eda.ipynb          # exploratory data analysis
├── src/
│   ├── fetch.py              # Adzuna API data fetching
│   └── db.py                 # PostgreSQL ingestion
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── .env                      # API credentials (not tracked)
├── .gitignore
└── README.md
```

---

## 🔧 Tech Stack

| Layer | Tools |
|-------|-------|
| Data Fetching | Adzuna Jobs API, Python `requests` |
| Storage | PostgreSQL, `psycopg2` |
| Analysis | `pandas`, `seaborn`, `matplotlib` |
| NLP | `spaCy` (lemmatization), `scikit-learn` TF-IDF |
| Modelling | XGBoost, scikit-learn |
| Dashboard | Streamlit |
| Environment | Python 3.10, conda |

---

## 📦 Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/job-market-pulse.git
cd job-market-pulse
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set up environment variables
Create a `.env` file in the project root:
```
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key
DB_USER=your_postgres_username
DB_PASSWORD=your_postgres_password
```

### 4. Set up PostgreSQL
```bash
brew install postgresql@15
brew services start postgresql@15
psql postgres -c "CREATE DATABASE job_market;"
```

Then create the jobs table using the schema in `src/db.py`.

### 5. Fetch data
```bash
python src/db.py
```

### 6. Run the dashboard
```bash
cd dashboard
streamlit run app.py
```

---

## 🔍 Methodology

### Data Collection
1,189 UK job postings fetched from the Adzuna Jobs API across 5 role categories:
- Data Scientist
- Data Analyst
- Data Engineer
- Machine Learning Engineer
- Business Intelligence Analyst

### Data Cleaning
- Removed zero and sub-£20k salary entries (day rates misreported as annual)
- Dropped rows with null company names
- Documented `company.average_salary` as an API field that returns null in practice

### Feature Engineering
- **Categorical encoding** — role, location, seniority, contract type/time via OrdinalEncoder
- **Seniority extraction** — derived from job title keywords (Junior/Mid/Senior/Lead)
- **Location simplification** — 189 unique locations grouped into 14 meaningful regions
- **Skill extraction** — regex word-boundary matching against a curated 31-skill taxonomy
- **TF-IDF** — lemmatized job descriptions vectorized, top 50 salary-correlated terms selected via SelectKBest

### Model
XGBoost Regressor trained on the combined engineered + NLP feature matrix.

| Metric | Score |
|--------|-------|
| CV R² (5-fold, shuffled) | 0.478 ± 0.023 |
| Test R² | 0.542 |
| MAE | £11,768 |

Cross validation used `KFold(shuffle=True)` — unshuffled folds produced a consistently negative R² on one fold due to data ordering artefacts, which shuffling resolved entirely.

The model explains ~48% of salary variance from job posting features alone. The remaining variance is attributed to factors unavailable in postings — candidate experience, company size, and negotiation.

---

## 🚀 Future Improvements

### 1. Larger Training Dataset
Current dataset of ~1,200 jobs limits model generalisation, particularly at salary extremes. Target 5,000+ postings by increasing API pagination, combining multiple job board sources, or scheduling weekly fetches to accumulate data over time.

### 2. Continual Training on Fresh Data
Salary benchmarks shift with the market. A scheduled pipeline (weekly cron job) to fetch new postings, retrain the model, and update the dashboard would keep predictions current. Dashboard trend visualisations could show how median salaries and in-demand skills have changed over time, with LLM-generated natural language summaries of key shifts.

### 3. Role-Specific Models
A single model trained across all five role types introduces noise — a data analyst and a machine learning engineer have fundamentally different salary drivers. Separate models per role would produce more accurate predictions, though this requires a larger dataset to maintain sufficient training examples per model.

### 4. Improved Salary Extremes
Tree-based models struggle to extrapolate beyond the range of training data, leading to underestimation of very high salaries (senior/principal roles, SC cleared positions) and overestimation at the low end. Quantile regression or separate contract vs permanent models could address this.

### 5. Company Background Data
Company-level features — funding stage (seed vs Series C vs public), headcount, and sector — are likely among the strongest salary predictors missing from the current model. These could be sourced via the Crunchbase API or Companies House public data.

### 6. Improved Skill Extraction
The current approach matches against a fixed 31-skill taxonomy, missing emerging tools and frameworks. Improvements include:
- **LLM-based extraction** — structured JSON skill extraction via API, adaptable to new technologies without manual list updates
- **NER model** — a fine-tuned spaCy NER model trained to recognise skill entities in job descriptions
- **Skills ontology** — dynamic skill lists sourced from maintained taxonomies such as ESCO or O*NET

### 7. Full Job Descriptions
Adzuna's search API truncates descriptions to the first paragraph. Full descriptions would provide richer NLP signal.