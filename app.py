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
import time

st.set_page_config(
    page_title="ResumeFlow • AI Resume Optimizer",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Eye-friendly Dark Theme with Smooth Animations
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0c0a1f 0%, #1a1833 100%);
        color: #e0e7ff;
    }
    .hero {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #5e72ff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 4s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 15px #5e72ff; }
        to { text-shadow: 0 0 30px #00d4ff; }
    }
    .tagline {
        text-align: center;
        color: #a5b4fc;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .glass-card {
        background: rgba(30, 30, 60, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(100, 120, 255, 0.2);
        border-radius: 18px;
        padding: 2rem;
        margin: 1.2rem 0;
        animation: slideUp 0.9s ease-out forwards;
        opacity: 0;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .score-circle {
        width: 195px;
        height: 195px;
        border-radius: 50%;
        margin: 25px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3.5rem;
        font-weight: 800;
        border: 10px solid;
        animation: pulse 2.5s infinite ease-in-out;
        box-shadow: 0 0 35px rgba(94, 114, 255, 0.5);
    }
    .high { border-color: #4ade80; color: #4ade80; }
    .medium { border-color: #facc15; color: #facc15; }
    .low { border-color: #fb7185; color: #fb7185; }
    .stButton>button {
        background: linear-gradient(90deg, #5e72ff, #00d4ff);
        color: white;
        border: none;
        border-radius: 50px;
        height: 3.6rem;
        font-size: 1.15rem;
        font-weight: 600;
        transition: all 0.4s ease;
    }
    .stButton>button:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 25px rgba(94, 114, 255, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="hero">ResumeFlow</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Smart AI that helps you land better opportunities</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/resume.png", width=85)
    st.title("ResumeFlow")
    st.markdown("**AI Resume Optimizer**")
    st.divider()
    st.markdown("**Features**")
    st.markdown("• Match Score with Animation\n• Tailored Resume\n• Cover Letter\n• Interview Questions\n• Smooth Animations")
    st.divider()
    st.caption("Eye-friendly design • Built for 2026 placements")

# API Key Setup
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found. Please add it in Streamlit Secrets.")
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

# Main Input
col1, col2 = st.columns([1, 1.2])

with col1:
    uploaded_file = st.file_uploader("📄 Upload Your Resume (PDF)", type="pdf")

with col2:
    jd_text = st.text_area("📝 Paste Job Description", height=160, placeholder="Paste the full job description here...")

if uploaded_file and jd_text:
    if st.button("🚀 Analyze Resume & Generate Full Application Kit", use_container_width=True):
        
        with st.spinner("AI is analyzing your profile..."):
            time.sleep(0.6)   # Small delay for better UX feel
            resume_text = extract_resume_text(uploaded_file)

            # Match Score Calculation
            resume_emb = embeddings.embed_query(resume_text)
            jd_emb = embeddings.embed_query(jd_text)
            match_score = round(cos_sim(torch.tensor([resume_emb]), torch.tensor([jd_emb]))[0][0].item() * 100, 1)

            score_class = "high" if match_score >= 75 else "medium" if match_score >= 55 else "low"
            feedback = "Excellent Match" if match_score >= 75 else "Good Potential" if match_score >= 55 else "Needs Improvement"

            # Celebration for high score
            if match_score >= 80:
                st.balloons()

            # Results in beautiful animated tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Match Score", 
                "💡 Suggestions", 
                "📝 Tailored Resume", 
                "✉️ Cover Letter", 
                "🎯 Interview Prep"
            ])

            with tab1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Resume Match Score")
                st.markdown(f"""
                <div class="score-circle {score_class}">
                    {match_score}%
                </div>
                <p style="text-align:center; font-size:1.25rem; margin-top:10px;">{feedback}</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                prompt = f"List missing skills and give 5 practical improvement suggestions.\nResume: {resume_text[:4000]}\nJob: {jd_text[:4000]}"
                response = llm.invoke(prompt).content
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                prompt = f"Rewrite the resume to perfectly match this job description. Keep it professional.\nResume: {resume_text[:4000]}\nJob: {jd_text[:4000]}"
                tailored = llm.invoke(prompt).content
                st.markdown(tailored)
                pdf_bytes = create_tailored_pdf(tailored)
                st.download_button("📥 Download Tailored Resume PDF", pdf_bytes, "Tailored_Resume.pdf", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab4:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                prompt = f"Write a compelling cover letter based on the resume and job.\nResume: {resume_text[:3000]}\nJob: {jd_text[:3000]}"
                cover = llm.invoke(prompt).content
                st.markdown(cover)
                pdf_cover = create_tailored_pdf(cover, "Cover Letter")
                st.download_button("📥 Download Cover Letter PDF", pdf_cover, "Cover_Letter.pdf", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab5:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                prompt = f"Generate 5 relevant interview questions with short sample answers.\nJob: {jd_text[:3000]}"
                interview = llm.invoke(prompt).content
                st.markdown(interview)
                st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:4rem 2rem;">
        <h3>Upload your resume and paste a job description</h3>
        <p>Get AI-powered insights, tailored resume, cover letter & interview preparation.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94a3b8; padding: 1rem;'>"
    "ResumeFlow • Eye-friendly AI Resume Optimizer with smooth animations"
    "</p>", 
    unsafe_allow_html=True
)
