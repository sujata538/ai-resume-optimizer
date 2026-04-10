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

st.set_page_config(
    page_title="ResumeFlow • AI Resume Optimizer",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Attractive Modern CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #f0f0ff;
    }
    .hero {
        text-align: center;
        padding: 3rem 0 2rem;
        background: linear-gradient(90deg, #7b5eff, #00f2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -2px;
    }
    .tagline {
        text-align: center;
        color: #b0b0ff;
        font-size: 1.25rem;
        margin-bottom: 2.5rem;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .score-circle {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        margin: 20px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3.8rem;
        font-weight: 900;
        border: 12px solid;
        box-shadow: 0 0 40px rgba(0, 255, 170, 0.4);
    }
    .high { border-color: #00ffaa; color: #00ffaa; }
    .medium { border-color: #ffcc00; color: #ffcc00; }
    .low { border-color: #ff4d94; color: #ff4d94; }
    .stButton>button {
        background: linear-gradient(90deg, #7b5eff, #00f2ff);
        color: white;
        border: none;
        border-radius: 50px;
        height: 3.5rem;
        font-size: 1.1rem;
        font-weight: 700;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(123, 94, 255, 0.5);
    }
    .tab-label {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown('<h1 class="hero">ResumeFlow</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Transform your resume into a job-winning masterpiece with AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/resume.png", width=90)
    st.title("ResumeFlow")
    st.markdown("### Your AI Career Co-Pilot")
    st.divider()
    st.markdown("**Features**")
    st.markdown("• Smart Match Score\n• Tailored Resume\n• Cover Letter\n• Interview Prep\n• Keyword Analysis")
    st.divider()
    st.caption("Made for ambitious students • 2026 Placements")

# API Setup
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please add GROQ_API_KEY in Streamlit Secrets")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data
def load_jobs():
    try:
        return pd.read_csv("jobs.csv")
    except:
        return pd.DataFrame({"title": ["Software Engineer"], "description": ["Python"]})

jobs_df = load_jobs()

def extract_resume_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page_text := page.extract_text():
                text += page_text + "\n"
    return text.strip()

def create_tailored_pdf(content, title="ResumeFlow Optimized"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    y = letter[1] - 70
    c.setFont("Helvetica-Bold", 16)
    c.drawString(70, y, title)
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

# Session State for History
if 'history' not in st.session_state:
    st.session_state.history = []

# Input Section
col_input1, col_input2 = st.columns([1, 1.3])

with col_input1:
    uploaded_file = st.file_uploader("📤 Upload Your Resume (PDF)", type="pdf", help="Max 5MB")

with col_input2:
    jd_text = st.text_area("📋 Paste Job Description", height=180, 
                          placeholder="Copy the full job description here...")

if uploaded_file and jd_text:
    if st.button("🚀 Generate My Complete Job Application Kit", use_container_width=True):
        
        with st.spinner("AI is working its magic..."):
            resume_text = extract_resume_text(uploaded_file)

            # Calculate Match Score
            resume_emb = embeddings.embed_query(resume_text)
            jd_emb = embeddings.embed_query(jd_text)
            match_score = round(cos_sim(torch.tensor([resume_emb]), torch.tensor([jd_emb]))[0][0].item() * 100, 1)

            score_class = "high" if match_score >= 75 else "medium" if match_score >= 55 else "low"
            feedback = "Outstanding Match 🔥" if match_score >= 75 else "Strong Potential 👍" if match_score >= 55 else "Room to Improve ⚡"

            # Save History
            st.session_state.history.append({"score": match_score, "jd": jd_text[:80] + "..."})
            if len(st.session_state.history) > 4:
                st.session_state.history.pop(0)

            # Beautiful Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Match Score", 
                "💡 Improvement Tips", 
                "📝 Tailored Resume", 
                "✉️ Cover Letter", 
                "🎯 Interview Prep"
            ])

            with tab1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Your Resume Match Score")
                st.markdown(f"""
                <div class="score-circle {score_class}">
                    {match_score}%
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**{feedback}**")
                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                prompt = f"Give specific improvement suggestions and list missing skills.\nResume: {resume_text[:4000]}\nJob: {jd_text[:4000]}"
                tips = llm.invoke(prompt).content
                st.markdown(tips)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                prompt = f"Rewrite this resume to strongly match the job. Use powerful language.\nResume: {resume_text[:4000]}\nJob: {jd_text[:4000]}"
                tailored = llm.invoke(prompt).content
                st.markdown(tailored)
                pdf_bytes = create_tailored_pdf(tailored)
                st.download_button("📥 Download Tailored Resume", pdf_bytes, "Tailored_Resume.pdf", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab4:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                prompt = f"Write a compelling, concise cover letter.\nResume: {resume_text[:3000]}\nJob: {jd_text[:3000]}"
                cover = llm.invoke(prompt).content
                st.markdown(cover)
                pdf_cover = create_tailored_pdf(cover, "Cover Letter")
                st.download_button("📥 Download Cover Letter", pdf_cover, "Cover_Letter.pdf", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab5:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                prompt = f"Generate 6 relevant interview questions with short sample answers for this role.\nJob: {jd_text[:3000]}"
                interview = llm.invoke(prompt).content
                st.markdown(interview)
                st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:4rem 2rem;">
        <h2>Ready to land your dream job?</h2>
        <p style="font-size:1.2rem;">Upload your resume and paste a job description to get a complete AI-powered application kit.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; padding:1rem;'>"
    "ResumeFlow • Beautiful AI Resume Optimizer • Built for 2026 Campus Placements"
    "</p>", 
    unsafe_allow_html=True
)
