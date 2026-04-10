import streamlit as st
import pdfplumber
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd
from sentence_transformers.util import cos_sim
import torch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Page Config - Must be first
st.set_page_config(
    page_title="AI Resume Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stButton>button {width: 100%; height: 3rem; background-color: #00c853; color: white; border: none;}
    .stButton>button:hover {background-color: #00b140;}
    .match-score {font-size: 2.5rem; font-weight: bold; text-align: center;}
    .high-score {color: #00c853;}
    .medium-score {color: #ff9800;}
    .low-score {color: #f44336;}
    </style>
""", unsafe_allow_html=True)

st.title("🚀 AI Resume Optimizer & Job Matcher")
st.markdown("**Upload your resume → Get AI-powered improvements & tailored resume**")

# Sidebar
with st.sidebar:
    st.header("📋 How to Use")
    st.markdown("""
    1. Upload your resume (PDF)
    2. Paste the Job Description
    3. Click Analyze
    4. Download tailored resume
    """)
    st.divider()
    st.caption("Built with Groq + LangChain + Streamlit\nPerfect for 2026 Placements")

# Load Groq API key safely
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found! Please add it in Streamlit Cloud → Secrets")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=groq_api_key
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load jobs with fallback
@st.cache_data
def load_jobs():
    try:
        return pd.read_csv("jobs.csv")
    except:
        data = {
            "title": ["Software Engineer", "Data Analyst", "ML Engineer"],
            "description": [
                "Python, Django, FastAPI, SQL, AWS",
                "Python, SQL, Power BI, Pandas",
                "Python, PyTorch, TensorFlow, NLP"
            ]
        }
        return pd.DataFrame(data)

jobs_df = load_jobs()

# Functions
def extract_resume_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def create_tailored_pdf(tailored_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 60
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "TAILORED RESUME")
    y -= 50
    
    c.setFont("Helvetica", 11)
    for line in tailored_text.split("\n"):
        if y < 50:
            c.showPage()
            y = height - 60
        c.drawString(50, y, line.strip())
        y -= 14
    c.save()
    buffer.seek(0)
    return buffer

# Main UI - Two Columns
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📄 Upload your Resume (PDF only)", type="pdf", help="Maximum 5MB")

with col2:
    jd_text = st.text_area("📝 Paste Job Description", height=250, 
                          placeholder="Paste the full job description here...")

if uploaded_file and jd_text:
    if st.button("🚀 Analyze Resume & Generate Tailored Version", type="primary"):
        with st.spinner("Analyzing your resume with AI..."):
            resume_text = extract_resume_text(uploaded_file)
            
            # Embeddings & Similarity
            resume_embedding = embeddings.embed_query(resume_text)
            jd_embedding = embeddings.embed_query(jd_text)
            similarity = cos_sim(torch.tensor([resume_embedding]), torch.tensor([jd_embedding]))[0][0].item()
            match_score = round(similarity * 100, 1)

            # Display Score with color
            score_color = "high-score" if match_score >= 75 else "medium-score" if match_score >= 50 else "low-score"
            st.markdown(f"<p class='match-score {score_color}'>{match_score}% Match</p>", unsafe_allow_html=True)

            # Sample Job Matches
            st.subheader("🔍 Top Matching Sample Jobs")
            job_embeddings = [embeddings.embed_query(desc) for desc in jobs_df['description']]
            similarities = [cos_sim(torch.tensor([resume_embedding]), torch.tensor([je]))[0][0].item() for je in job_embeddings]
            jobs_df['score'] = [round(s*100, 1) for s in similarities]
            st.dataframe(jobs_df.nlargest(3, 'score')[['title', 'score']], hide_index=True)

            # AI Analysis
            prompt = f"""
            You are an expert Indian career coach helping students from Uttar Pradesh get better placements.
            
            Resume: {resume_text[:4000]}
            Job Description: {jd_text[:4000]}
            
            Provide in this exact format:
            MISSING SKILLS: (list 5-7 important keywords from JD missing in resume)
            IMPROVEMENT SUGGESTIONS: (4-5 specific bullet point changes)
            TAILORED RESUME: (full rewritten resume optimized for this JD, keep professional tone)
            """

            response = llm.invoke(prompt)
            result = response.content

            st.subheader("✨ AI Analysis & Recommendations")
            st.markdown(result)

            # Extract and offer download
            if "TAILORED RESUME:" in result:
                tailored_part = result.split("TAILORED RESUME:")[-1].strip()
            else:
                tailored_part = result

            pdf_bytes = create_tailored_pdf(tailored_part)

            st.download_button(
                label="📥 Download Tailored Resume (PDF)",
                data=pdf_bytes,
                file_name="Tailored_Resume.pdf",
                mime="application/pdf",
                type="primary"
            )

            st.success("✅ Analysis Complete! Use the suggestions to improve your chances.")

else:
    st.info("👆 Upload your resume and paste a Job Description to get started.")

st.caption("Made with ❤️ for Uttar Pradesh students | Groq + LangChain + Streamlit")
