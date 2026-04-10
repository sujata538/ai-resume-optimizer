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
import re

st.set_page_config(page_title="ResumeFlow • AI Optimizer", page_icon="✦", layout="centered")

# Unique Minimalist CSS (kept from last version)
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #0a0a1f 0%, #1a1a2e 100%); color: #e0e0ff;}
    .header {text-align: center; padding: 2rem 0 1rem; background: linear-gradient(90deg, #6b4eff, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 700;}
    .tagline {text-align: center; color: #a0a0cc; font-size: 1.1rem; margin-bottom: 2rem;}
    .card {background: rgba(255,255,255,0.06); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 1.8rem; margin: 1rem 0;}
    .score-circle {width: 170px; height: 170px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 15px auto; font-size: 3rem; font-weight: 700; border: 8px solid;}
    .high {border-color: #00ffaa; color: #00ffaa;}
    .medium {border-color: #ffd700; color: #ffd700;}
    .low {border-color: #ff4d4d; color: #ff4d4d;}
    .stButton>button {background: linear-gradient(90deg, #6b4eff, #00d4ff); color: white; border: none; border-radius: 50px; font-weight: 600; width: 100%;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header">ResumeFlow</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">AI that turns good resumes into great ones</p>', unsafe_allow_html=True)

# API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY missing → Add in Streamlit Secrets")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data
def load_jobs():
    try:
        return pd.read_csv("jobs.csv")
    except:
        return pd.DataFrame({"title": ["Software Engineer"], "description": ["Python, SQL"]})

jobs_df = load_jobs()

def extract_resume_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page_text := page.extract_text():
                text += page_text + "\n"
    return text.strip()

def create_tailored_pdf(content, title="ResumeFlow Optimized Resume"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    y = letter[1] - 70
    c.setFont("Helvetica-Bold", 16)
    c.drawString(70, y, title.upper())
    y -= 50
    c.setFont("Helvetica", 11)
    for line in content.splitlines():
        if y < 70:
            c.showPage()
            y = letter[1] - 70
        c.drawString(70, y, line.strip())
        y -= 15
    c.save()
    buffer.seek(0)
    return buffer

# Session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Main Inputs
st.markdown("### Upload & Target")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Resume (PDF)", type="pdf")
with col2:
    jd_text = st.text_area("Job Description", height=160, placeholder="Paste full JD here...")

if uploaded_file and jd_text:
    if st.button("✦ Analyze, Optimize & Generate Full Kit", use_container_width=True):
        with st.spinner("Analyzing with AI..."):
            resume_text = extract_resume_text(uploaded_file)

            # Core Match Score
            resume_emb = embeddings.embed_query(resume_text)
            jd_emb = embeddings.embed_query(jd_text)
            match_score = round(cos_sim(torch.tensor([resume_emb]), torch.tensor([jd_emb]))[0][0].item() * 100, 1)

            score_class = "high" if match_score >= 75 else "medium" if match_score >= 55 else "low"
            feedback = "Excellent Match" if match_score >= 75 else "Good Potential" if match_score >= 55 else "Needs Strong Improvement"

            # Save to history
            st.session_state.history.append({
                "score": match_score,
                "jd": jd_text[:100] + "..."
            })
            if len(st.session_state.history) > 3:
                st.session_state.history.pop(0)

            # Tabs for organized output
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Score & Keywords", "💡 Suggestions", "📝 Tailored Resume", "✉️ Cover Letter", "❓ Interview Prep"])

            with tab1:
                st.markdown(f"""
                <div class="card" style="text-align:center">
                    <div class="score-circle {score_class}">{match_score}%</div>
                    <p style="font-size:1.2rem;">{feedback}</p>
                </div>
                """, unsafe_allow_html=True)

                # Simple Keyword Analysis
                st.subheader("Keyword Insights")
                jd_lower = jd_text.lower()
                common_words = re.findall(r'\b\w+\b', jd_lower)
                top_keywords = pd.Series(common_words).value_counts().head(8)
                st.write("**Top keywords in JD:**", ", ".join(top_keywords.index.tolist()))

            with tab2:
                prompt = f"Resume: {resume_text[:4000]}\nJob: {jd_text[:4000]}\nList missing critical skills and give 5 concrete improvement suggestions."
                suggestions = llm.invoke(prompt).content
                st.markdown(suggestions)

            with tab3:
                prompt = f"Rewrite the resume to perfectly match this job. Be concise, use strong action verbs.\nResume: {resume_text[:4000]}\nJob: {jd_text[:4000]}"
                tailored = llm.invoke(prompt).content
                st.markdown(tailored)

                pdf_bytes = create_tailored_pdf(tailored)
                st.download_button("Download Tailored Resume PDF", pdf_bytes, "ResumeFlow_Tailored_Resume.pdf", "application/pdf", use_container_width=True)

            with tab4:
                prompt = f"Write a professional, concise cover letter tailored to this job using the resume info.\nResume: {resume_text[:3000]}\nJob: {jd_text[:3000]}"
                cover_letter = llm.invoke(prompt).content
                st.markdown(cover_letter)

                pdf_cover = create_tailored_pdf(cover_letter, "ResumeFlow Cover Letter")
                st.download_button("Download Cover Letter PDF", pdf_cover, "ResumeFlow_Cover_Letter.pdf", "application/pdf", use_container_width=True)

            with tab5:
                prompt = f"Generate 5 strong interview questions for this role based on the job description and resume.\nJob: {jd_text[:3000]}"
                questions = llm.invoke(prompt).content
                st.markdown(questions)

            # History Sidebar (collapsed by default)
            with st.sidebar:
                st.header("Recent Analyses")
                for i, entry in enumerate(reversed(st.session_state.history)):
                    st.caption(f"{i+1}. {entry['score']}% — {entry['jd']}")

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem 1rem;">
        <h3>Ready to stand out?</h3>
        <p>Upload your resume + paste a Job Description to get a full AI application kit.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#777; font-size:0.9rem;'>ResumeFlow • Minimal AI Resume Optimizer with Full Application Kit</p>", unsafe_allow_html=True)
