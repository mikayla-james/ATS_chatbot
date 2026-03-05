"""
ATS Resume Chatbot - Open Source Job Application Assistant
Built with Streamlit + Hugging Face Transformers + Google Sheets

Features:
1. Score resume vs job description (cosine similarity)
2. Auto-extract keywords from job posts
3. Suggest resume tweaks per job
4. Read/write to Google Sheets job tracker
"""

import streamlit as st
import re
import os
from typing import Optional
from datetime import datetime

# ---------------------------------------------------------------------------
# Google Sheets config
# ---------------------------------------------------------------------------
WORKBOOK_NAME = "Job Applications Tracker"
SHEET_NAME = "Tracker"
CREDENTIALS_FILE = "credentials.json"

st.set_page_config(
    page_title="ATS Resume Chatbot",
    page_icon="target",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Google Sheets connection
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Google Sheets...")
def connect_to_sheets():
    try:
        import gspread
        creds_path = os.path.join(os.path.dirname(__file__), CREDENTIALS_FILE)
        gc = gspread.service_account(filename=creds_path)
        workbook = gc.open(WORKBOOK_NAME)
        sheet = workbook.worksheet(SHEET_NAME)
        return sheet
    except FileNotFoundError:
        st.warning("credentials.json not found. Google Sheets features are disabled. "
                   "Place your service account JSON in the same folder as this app.")
        return None
    except Exception as e:
        st.warning(f"Could not connect to Google Sheets: {e}")
        return None


def log_to_sheet(sheet, company, title, score, missing_keywords):
    if sheet is None:
        return False
    try:
        headers = sheet.row_values(1)
        header_map = {h.strip().lower(): i + 1 for i, h in enumerate(headers)}
        all_values = sheet.col_values(1)
        next_row = len(all_values) + 1
        row_data = [""] * len(headers)

        if "company" in header_map:
            row_data[header_map["company"] - 1] = company
        if "title" in header_map:
            row_data[header_map["title"] - 1] = title
        if "status" in header_map:
            row_data[header_map["status"] - 1] = "Pending"
        if "application date" in header_map:
            row_data[header_map["application date"] - 1] = datetime.now().strftime("%m/%d/%Y")
        if "notes" in header_map:
            missing_str = ", ".join(missing_keywords[:10])
            row_data[header_map["notes"] - 1] = f"Match: {score}% | Missing: {missing_str}"

        sheet.append_row(row_data, value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        st.error(f"Failed to log to sheet: {e}")
        return False


def read_past_scores(sheet):
    if sheet is None:
        return []
    try:
        records = sheet.get_all_records()
        scored = []
        for row in records:
            notes = str(row.get("Notes", ""))
            if "Match:" in notes:
                scored.append({
                    "company": row.get("Company", ""),
                    "title": row.get("Title", ""),
                    "date": row.get("Application Date", ""),
                    "notes": notes,
                })
        return scored
    except Exception as e:
        st.error(f"Failed to read from sheet: {e}")
        return []


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models - this takes a minute the first time...")
def load_models():
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    generator = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=300,
        return_full_text=False,
    )
    return embedder, generator


# ---------------------------------------------------------------------------
# Core ATS functions
# ---------------------------------------------------------------------------
def compute_similarity(embedder, text_a, text_b):
    embeddings = embedder.encode([text_a, text_b])
    from numpy import dot
    from numpy.linalg import norm
    cos_sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
    return float(cos_sim)


def extract_keywords(job_description):
    from collections import Counter

    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need", "must",
        "that", "this", "these", "those", "it", "its", "you", "your", "we",
        "our", "they", "their", "them", "he", "she", "him", "her", "who",
        "which", "what", "where", "when", "how", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "no", "not",
        "only", "own", "same", "so", "than", "too", "very", "just", "about",
        "above", "after", "again", "also", "as", "before", "between", "during",
        "into", "through", "under", "until", "up", "out", "over", "if", "then",
        "once", "here", "there", "why", "any", "able", "etc", "including",
        "well", "across", "within", "while", "using", "based", "related",
        "strong", "new", "work", "working", "experience", "role", "team",
        "ability", "skills", "join", "looking", "ideal", "candidate",
        "responsibilities", "requirements", "qualifications", "preferred",
        "required", "minimum", "plus", "years", "year", "position", "company",
    }

    known_phrases = [
        "machine learning", "deep learning", "natural language processing",
        "computer vision", "data science", "data engineering", "data analysis",
        "data visualization", "feature engineering", "model deployment",
        "a/b testing", "ab testing", "ci/cd", "ci cd", "etl pipeline",
        "data pipeline", "cloud computing", "project management",
        "agile methodology", "cross functional", "cross-functional",
        "stakeholder management", "business intelligence", "power bi",
        "time series", "reinforcement learning", "large language models",
        "generative ai", "prompt engineering", "retrieval augmented generation",
        "rag", "mlops", "devops", "full stack", "front end", "back end",
        "rest api", "graphql", "microservices", "object oriented",
        "version control", "unit testing", "statistical analysis",
        "regression analysis", "hypothesis testing", "experimental design",
        "random forest", "gradient boosting", "neural network",
        "convolutional neural network", "recurrent neural network",
        "transfer learning", "fine tuning", "fine-tuning",
    ]

    tech_terms = {
        "python", "sql", "r", "java", "scala", "javascript", "typescript",
        "c++", "c#", "rust", "go", "julia", "matlab", "sas", "spss",
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "spark", "hadoop", "hive", "kafka", "airflow", "dbt",
        "docker", "kubernetes", "terraform", "jenkins", "git", "github",
        "aws", "azure", "gcp", "snowflake", "redshift", "bigquery",
        "databricks", "sagemaker", "mlflow", "wandb",
        "tableau", "looker", "powerbi", "excel", "domo",
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "neo4j", "cassandra", "dynamodb",
        "flask", "fastapi", "django", "streamlit", "gradio",
        "bert", "gpt", "llm", "llms", "transformers", "huggingface",
        "langchain", "openai", "anthropic",
        "xgboost", "lightgbm", "catboost", "shap", "lime",
        "opencv", "pillow", "spacy", "nltk", "gensim",
        "linux", "bash", "slurm", "hpc",
        "jira", "confluence", "notion", "asana",
        "pmp", "scrum", "kanban",
    }

    jd_lower = job_description.lower()

    found_phrases = []
    for phrase in known_phrases:
        if phrase in jd_lower:
            found_phrases.append(phrase.title())

    words = re.findall(r"[a-z#+.]+", jd_lower)
    found_tech = []
    for w in set(words):
        if w in tech_terms:
            found_tech.append(w if w.isupper() or len(w) <= 3 else w.title())

    acronym_map = {
        "Aws": "AWS", "Gcp": "GCP", "Sql": "SQL", "Hpc": "HPC",
        "Etl": "ETL", "Ci/Cd": "CI/CD", "Mlops": "MLOps", "Devops": "DevOps",
        "Llm": "LLM", "Llms": "LLMs", "Rag": "RAG", "Bert": "BERT",
        "Gpt": "GPT", "R": "R", "Sas": "SAS", "Spss": "SPSS",
        "Api": "API", "Rest Api": "REST API",
    }
    found_phrases = [acronym_map.get(p, p) for p in found_phrases]
    found_tech = [acronym_map.get(t, t) for t in found_tech]

    word_counts = Counter(words)
    domain_keywords = []
    for word, count in word_counts.items():
        if count >= 2 and word not in stop_words and word not in tech_terms and len(word) > 3:
            domain_keywords.append(word.title())

    return {
        "technical_skills": sorted(set(found_tech)),
        "key_phrases": sorted(set(found_phrases)),
        "domain_keywords": sorted(set(domain_keywords[:15])),
    }


def find_missing_keywords(resume_text, keywords):
    resume_lower = resume_text.lower()
    missing = {"technical_skills": [], "key_phrases": [], "domain_keywords": []}
    present = {"technical_skills": [], "key_phrases": [], "domain_keywords": []}

    for category, terms in keywords.items():
        for term in terms:
            if term.lower() in resume_lower:
                present[category].append(term)
            else:
                missing[category].append(term)

    return {"present": present, "missing": missing}


def generate_suggestions(generator, resume_text, job_description, missing):
    missing_flat = []
    for cat, terms in missing.items():
        missing_flat.extend(terms)

    if not missing_flat:
        return "Your resume already covers the key terms in this job description. Focus on quantifying your impact with numbers and metrics."

    missing_str = ", ".join(missing_flat[:15])

    prompt = f"""<|system|>
You are an expert resume coach. Give specific, actionable advice.</s>
<|user|>
My resume is missing these keywords from a job description I'm applying to: {missing_str}

Give me 3-5 specific suggestions for how to naturally add these to my resume. Be concrete about what to write and where to put it.</s>
<|assistant|>"""

    result = generator(prompt, max_new_tokens=300)
    return result[0]["generated_text"]


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    sheet = connect_to_sheets()

    with st.sidebar:
        st.title("ATS Resume Chatbot")
        st.markdown("---")

        if sheet:
            st.success("Connected to Google Sheets")
        else:
            st.warning("Sheets not connected")

        st.markdown(
            """
            **Open-source ATS assistant**

            Paste your resume and a job description
            to get instant feedback:

            - **Match Score** - cosine similarity
            - **Keyword Extraction** - what the ATS looks for
            - **Resume Tweaks** - specific suggestions

            ---
            *Built with Streamlit + HuggingFace*
            """
        )

        if sheet:
            st.markdown("---")
            st.markdown("**Past ATS Scores**")
            past = read_past_scores(sheet)
            if past:
                for entry in reversed(past[-10:]):
                    st.markdown(f"**{entry['company']}** - {entry['title']}")
                    st.caption(f"{entry['date']} | {entry['notes']}")
            else:
                st.caption("No scored applications yet.")

        st.markdown("---")
        st.markdown("**How it works**")
        st.markdown(
            """
            1. Paste your resume text
            2. Paste a job description
            3. Click **Analyze**
            4. Get your score + suggestions
            """
        )

    st.title("ATS Resume Analyzer")
    st.markdown("Paste your resume and a job description below to see how well you match and get tailored suggestions.")

    if sheet:
        log_col1, log_col2 = st.columns(2)
        with log_col1:
            company_name = st.text_input("Company name", placeholder="e.g. Google")
        with log_col2:
            job_title = st.text_input("Job title", placeholder="e.g. Data Scientist")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Resume")
        resume_text = st.text_area(
            "Paste your resume text here",
            height=350,
            placeholder="Copy and paste the full text of your resume...",
            key="resume",
        )

    with col2:
        st.subheader("Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=350,
            placeholder="Copy and paste the full job description...",
            key="jd",
        )

    analyze_btn = st.button("Analyze Match", type="primary", use_container_width=True)

    if analyze_btn:
        if not resume_text.strip() or not job_description.strip():
            st.warning("Please paste both your resume and a job description.")
            return

        embedder, generator = load_models()

        with st.spinner("Calculating match score..."):
            score = compute_similarity(embedder, resume_text, job_description)
            score_pct = round(score * 100, 1)

        st.markdown("---")

        score_col, detail_col = st.columns([1, 2])
        with score_col:
            st.metric("Match Score", f"{score_pct}%")
            if score_pct >= 75:
                st.success("Strong match!")
            elif score_pct >= 55:
                st.info("Decent match - some tweaks could help.")
            else:
                st.warning("Low match - consider tailoring your resume more.")

        with st.spinner("Extracting ATS keywords..."):
            keywords = extract_keywords(job_description)
            gaps = find_missing_keywords(resume_text, keywords)

        with detail_col:
            st.subheader("Keyword Analysis")

            tab1, tab2 = st.tabs(["Keywords You Have", "Keywords You're Missing"])

            with tab1:
                for category, terms in gaps["present"].items():
                    if terms:
                        label = category.replace("_", " ").title()
                        st.markdown(f"**{label}:** {', '.join(terms)}")
                if not any(gaps["present"].values()):
                    st.markdown("*No matching keywords found - your resume may need significant tailoring.*")

            with tab2:
                for category, terms in gaps["missing"].items():
                    if terms:
                        label = category.replace("_", " ").title()
                        st.markdown(f"**{label}:** {', '.join(terms)}")
                if not any(gaps["missing"].values()):
                    st.markdown("*Great - no major keyword gaps detected!*")

        st.markdown("---")
        st.subheader("Resume Improvement Suggestions")

        with st.spinner("Generating tailored suggestions..."):
            suggestions = generate_suggestions(generator, resume_text, job_description, gaps["missing"])

        st.markdown(suggestions)

        st.markdown("---")
        st.subheader("Quick Stats")
        stat_cols = st.columns(4)
        total_present = sum(len(v) for v in gaps["present"].values())
        total_missing = sum(len(v) for v in gaps["missing"].values())
        total_kw = total_present + total_missing

        stat_cols[0].metric("Total Keywords Found", len(keywords["technical_skills"]) + len(keywords["key_phrases"]))
        stat_cols[1].metric("You Have", total_present)
        stat_cols[2].metric("You're Missing", total_missing)
        stat_cols[3].metric("Coverage", f"{round(total_present / max(total_kw, 1) * 100)}%")

        if sheet:
            st.markdown("---")
            if company_name.strip() and job_title.strip():
                all_missing = []
                for terms in gaps["missing"].values():
                    all_missing.extend(terms)

                if st.button("Save to Google Sheets", type="secondary", use_container_width=True):
                    success = log_to_sheet(sheet, company_name, job_title, score_pct, all_missing)
                    if success:
                        st.success(f"Logged {company_name} - {job_title} to your tracker!")
                        st.cache_resource.clear()
                    else:
                        st.error("Failed to log. Check your credentials and sheet permissions.")
            else:
                st.info("Enter a company name and job title above to save results to your tracker.")

    st.markdown("---")
    st.subheader("Ask Me Anything")
    st.markdown("Have questions about your resume, ATS systems, or job applications? Ask below.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your resume, ATS tips, interview prep..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        embedder, generator = load_models()

        context = ""
        if resume_text.strip():
            context += f"The user's resume mentions: {resume_text[:500]}... "
        if job_description.strip():
            context += f"They're applying to a job that requires: {job_description[:500]}... "

        chat_prompt = f"""<|system|>
You are a helpful resume and career coach. Be concise and actionable.</s>
<|user|>
{context}

{prompt}</s>
<|assistant|>"""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generator(chat_prompt, max_new_tokens=300)
                reply = response[0]["generated_text"]
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()