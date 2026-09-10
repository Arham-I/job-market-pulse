import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import joblib
import psycopg2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from scipy.sparse import hstack, csr_matrix
import spacy

load_dotenv()

# --- Load Models ---
model = joblib.load('../models/model.pkl')
tfidf = joblib.load('../models/tfidf.pkl')
selector = joblib.load('../models/selector.pkl')
# le = joblib.load('../models/label_encoder.pkl')
oe = joblib.load('../models/ordinal_encoder.pkl')
skills_cols = joblib.load('../models/skill_cols.pkl')

# --- Load spaCy ---
nlp = spacy.load('en_core_web_sm')

# --- DB Connection ---
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        dbname="job_market",
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
        # host="localhost",
        # port="5432"
    )

# --- Load Data ---
@st.cache_data
def load_data():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM jobs;", conn)

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Job Market Pulse")
page = st.sidebar.radio("Navigate", ["Market Overview", "Salary Predictor"])

if page == "Market Overview":
    st.title("🇬🇧 UK DS/ML Job Market Overview")

    # --- Data Prep ---
    df = df[df['salary_min'] > 0]
    df = df[df['salary_max'] >= 20000]
    df['salary_mid'] = (df['salary_min'] + df['salary_max']) / 2
    df = df[df['company'].notna()]

    def simplify_location(loc):
        loc = loc.lower()
        if 'london' in loc: return 'London'
        elif 'manchester' in loc: return 'Manchester'
        elif 'birmingham' in loc: return 'Birmingham'
        elif 'leeds' in loc: return 'Leeds'
        elif 'bristol' in loc: return 'Bristol'
        elif 'edinburgh' in loc or 'glasgow' in loc or 'scotland' in loc: return 'Scotland'
        elif 'belfast' in loc or 'northern ireland' in loc: return 'Northern Ireland'
        elif 'newcastle' in loc: return 'Newcastle'
        elif 'sheffield' in loc: return 'Sheffield'
        elif 'liverpool' in loc: return 'Liverpool'
        elif 'cheltenham' in loc or 'gloucester' in loc: return 'Gloucestershire'
        elif 'cambridge' in loc: return 'Cambridge'
        elif 'milton keynes' in loc: return 'Milton Keynes'
        else: return 'Other'

    def extract_seniority(title):
        title = title.lower()
        if any(word in title for word in ['head', 'director', 'vp', 'vice president', 'principal', 'lead']):
            return 'Lead'
        elif any(word in title for word in ['senior', 'sr.', 'sr ']):
            return 'Senior'
        elif any(word in title for word in ['junior', 'jr.', 'jr ', 'graduate', 'intern', 'entry']):
            return 'Junior'
        else:
            return 'Mid'

    df['location_simplified'] = df['location'].apply(simplify_location)
    df['seniority'] = df['title'].apply(extract_seniority)

     # --- Row 1: Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs", f"{len(df):,}")
    col2.metric("Median Salary", f"£{df['salary_mid'].median():,.0f}")
    col3.metric("Avg Salary", f"£{df['salary_mid'].mean():,.0f}")
    col4.metric("Locations", df['location_simplified'].nunique())

    st.markdown("---")

    # --- Row 2: Salary by Role ---
    st.subheader("Salary by Role")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='search_term', y='salary_mid', ax=ax1)
    ax1.set_xlabel('Role')
    ax1.set_ylabel('Mid Salary (£)')
    ax1.tick_params(axis='x', rotation=15)
    st.pyplot(fig1)

    st.markdown("---")

    # --- Row 3: Salary by Location ---
    st.subheader("Salary by Location")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    order = df.groupby('location_simplified')['salary_mid'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='location_simplified', y='salary_mid', order=order, ax=ax2)
    ax2.set_xlabel('Location')
    ax2.set_ylabel('Mid Salary (£)')
    ax2.tick_params(axis='x', rotation=30)
    st.pyplot(fig2)

    st.markdown("---")

    # --- Row 4: Job Count by Seniority ---
    st.subheader("Job Count by Seniority")

    fig3, ax3 = plt.subplots(1, 2, figsize=(16, 6))

    # Count by seniority
    sns.countplot(data=df, x='seniority', 
                order=['Junior', 'Mid', 'Senior', 'Lead'], 
                ax=ax3[0])
    ax3[0].set_title('Job Count by Seniority')
    ax3[0].set_xlabel('Seniority Level')
    ax3[0].set_ylabel('Count')

    # Salary by seniority
    sns.boxplot(data=df, x='seniority', y='salary_mid',
                order=['Junior', 'Mid', 'Senior', 'Lead'],
                ax=ax3[1])
    ax3[1].set_title('Salary by Seniority Level')
    ax3[1].set_xlabel('Seniority Level')
    ax3[1].set_ylabel('Mid Salary (£)')

    plt.suptitle('Seniority Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("---")

    # --- Row 5: Top Skills ---
    st.subheader("High Demand Skills in the UK Job Market")

    skills = [
        # General
        'python', 'sql', 'r', 'scala', 'java', 'git', 'linux', 'excel',
        # Data Science
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 'matplotlib', 'seaborn', 'jupyter', 'statsmodels',
        # ML/DL
        'pytorch', 'tensorflow', 'keras', 'mlflow', 'hugging face', 'fastapi',
        # Data Engineering
        'spark', 'hadoop', 'kafka', 'airflow', 'docker', 'kubernetes', 'dbt',
        # Cloud
        'aws', 'gcp', 'azure',
        # BI
        'power bi', 'tableau', 'looker', 'dax', 'qlik',
    ]

    soft_skills = [
        'communication', 'stakeholder', 'collaboration', 'teamwork',
        'problem solving', 'critical thinking', 'presentation',
        'leadership', 'agile', 'scrum', 'project management',
        'analytical', 'attention to detail', 'self-starter', 'initiative'
    ]

    df["description"] = df["description"].str.lower()

    def extract_skills(description, skill_list):
        found = []
        for skill in skill_list:
            # Use word boundary matching for single words
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, description):
                found.append(skill)
        return found

    df["skills_found"] = df["description"].apply(lambda x: extract_skills(x, skills))
    df["soft_skills_found"] = df["description"].apply(lambda x: extract_skills(x, soft_skills))

    # Count both
    all_hard = [skill for sublist in df["skills_found"] for skill in sublist]
    all_soft = [skill for sublist in df["soft_skills_found"] for skill in sublist]

    hard_df = pd.DataFrame(Counter(all_hard).most_common(20), columns=["skill", "count"])
    soft_df = pd.DataFrame(Counter(all_soft).most_common(15), columns=["skill", "count"])

    print(hard_df)
    print(soft_df)

    # Side by side figure
    fig4, ax4 = plt.subplots(1, 2, figsize=(18, 7))

    sns.barplot(data=hard_df, x='count', y='skill', ax=ax4[0])
    ax4[0].set_title('Most In-Demand Technical Skills', fontsize=20)
    ax4[0].set_xlabel('Number of Job Postings')
    ax4[0].set_ylabel('Skill')

    sns.barplot(data=soft_df, x='count', y='skill', ax=ax4[1])
    ax4[1].set_title('Most In-Demand Soft Skills', fontsize=20)
    ax4[1].set_xlabel('Number of Job Postings')
    ax4[1].set_ylabel('')

    plt.tight_layout()
    st.pyplot(fig4)

    
    
elif page == "Salary Predictor":
    st.title("Salary Predictor")
    st.markdown("Enter job details below to predict the expected salary.")

    # --- Input Widgets ---
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.selectbox("Role", [
            "data scientist", "data analyst", "data engineer",
            "machine learning engineer", "business intelligence analyst"
        ])
        location = st.selectbox("Location", [
            "London", "Manchester", "Birmingham", "Leeds", "Bristol",
            "Scotland", "Northern Ireland", "Newcastle", "Sheffield",
            "Liverpool", "Gloucestershire", "Cambridge", "Milton Keynes",
            "Other"
        ])
        seniority = st.selectbox("Seniority", ["Junior", "Mid", "Senior", "Lead"])

    with col2:
        contract_type = st.selectbox("Contract Type", ["permanent", "contract", "unknown"])
        contract_time = st.selectbox("Contract Time", ["full_time", "part_time", "unknown"])
        salary_is_predicted = st.checkbox("Salary is estimated (not employer stated)", value=True)

    description = st.text_area("Job Description", height=200, 
                                placeholder="Paste the job description here...")

    if st.button("Predict Salary"):
            if not description:
                st.warning("Please enter a job description.")
            else:
                # --- Feature Engineering ---
                # search_term_enc = le.transform([search_term])[0]
                # location_enc = le.transform([location])[0]
                # seniority_enc = le.transform([seniority])[0]
                # contract_type_enc = le.transform([contract_type])[0]
                # contract_time_enc = le.transform([contract_time])[0]

                encoded = list(oe.transform([[search_term, location, seniority, contract_type, contract_time]])[0])

                # --- Extract Skills from Description ---
                description = description.lower()

                def extract_skills(description, skill_list):
                    found = []
                    for skill in skill_list:
                        # Use word boundary matching for single words
                        pattern = r'\b' + re.escape(skill) + r'\b'
                        if re.search(pattern, description):
                            found.append(skill)
                    return found

                found_skills = extract_skills(description, skills_cols)
                skill_values = [1 if skill in found_skills else 0 for skill in skills_cols]

                # --- Lemmatize description ---
                doc = nlp(description)
                lemmatized = ' '.join([token.lemma_ for token in doc if not token.is_punct])

                # --- TF-IDF + Selection ---
                tfidf_vec = tfidf.transform([lemmatized])
                tfidf_selected = selector.transform(tfidf_vec)

                # --- Assemble Feature Row ---
                engineered = csr_matrix([encoded + 
                        [int(salary_is_predicted)] 
                        + skill_values
                    ])

                X_input = hstack([engineered, tfidf_selected])

                # --- Predict ---
                prediction = model.predict(X_input)[0]
                st.success(f"### Predicted Salary: £{prediction:,.0f}")
