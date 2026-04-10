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

# Minimalist & Unique Page Config
st.set_page_config(
    page_title="ResumeFlow • AI Optimizer",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Unique Minimalist CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0a0a1f 0%, #1a1a2e 100%);
        color: #e0e0ff;
    }
    .header {
        text-align: center;
        padding: 2rem 0 1.5rem;
        background: linear-gradient(90deg, #6b4eff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
    }
    .tagline {
        text-align: center;
        color: #a0a0cc;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
    }
    .score-circle {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 20px auto;
        font-size: 3.2rem;
        font-weight: 700;
        border: 8px solid;
        background: rgba(0,0,0,0.3);
    }
    .high { border-color: #00ffaa; color: #00ffaa; }
    .medium { border-color: #ffd700; color: #ffd700; }
    .low { border-color: #ff4d4d; color: #ff4d4d; }
    .stButton>button {
        background: linear-gradient(90deg, #6b4eff, #00d4ff);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header">ResumeFlow</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">AI that turns good resumes into great ones</p>', unsafe_allow_html=True)

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
        return pd.DataFrame({"title": ["Software Engineer"], "description": ["Python, SQL, Git"]})

jobs_df = load_jobs()

def extract_resume_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page_text := page.extract_text():
                text += page_text + "\n"
    return text.strip()

def create_tailored_pdf(tailored_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    y = letter[1] - 70
    c.setFont("Helvetica-Bold", 18)
    c.drawString(70, y, "RESUMEFLOW - AI TAILORED RESUME")
    y -= 50
    c.setFont("Helvetica", 11)
    for line in tailored_text.splitlines():
        if y < 70:
            c.showPage()
            y = letter[1] - 70
        c.drawString(70, y, line.strip())
        y -= 15
    c.save()
    buffer.seek(0)
    return buffer

# Upload Section
st.markdown("### Upload your resume and target job")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Resume (PDF)", type="pdf", label_visibility="collapsed")

with col2:
    jd_text = st.text_area("Job Description", height=180, placeholder="Paste the full JD here...", label_visibility="collapsed")

if uploaded_file and jd_text:
    if st.button("✦ Analyze & Optimize Resume", use_container_width=True):
        
        with st.spinner("Thinking deeply..."):
            resume_text = extract_resume_text(uploaded_file)

            # Match Score
            resume_emb = embeddings.embed_query(resume_text)
            jd_emb = embeddings.embed_query(jd_text)
            score = round(cos_sim(torch.tensor([resume_emb]), torch.tensor([jd_emb]))[0][0].item() * 100, 1)

            if score >= 75:
                score_class = "high"
                feedback = "Excellent alignment"
            elif score >= 55:
                score_class = "medium"
                feedback = "Good potential"
            else:
                score_class = "low"
                feedback = "Significant improvements needed"

            # Results in clean cards
            st.markdown("### Analysis Result")

            col_a, col_b = st.columns([1, 2])

            with col_a:
                st.markdown(f"""
                <div class="card" style="text-align:center">
                    <div class="score-circle {score_class}">
                        {score}%
                    </div>
                    <p style="margin-top:10px; font-size:1.1rem;">{feedback}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("🔑 Missing Key Skills")
                
                prompt = f"""
                Resume: {resume_text[:3500]}
                Job: {jd_text[:3500]}
                List only the most important missing skills from the JD (maximum 6 bullet points).
                """
                missing = llm.invoke(prompt).content
                st.markdown(missing)
                st.markdown('</div>', unsafe_allow_html=True)

            # Tailored Resume Section
            st.markdown("### 📄 Your AI-Optimized Resume")

            full_prompt = f"""
            Rewrite this resume to strongly match the job description.
            Use powerful action verbs. Keep it concise and professional.
            Original Resume: {resume_text[:4000]}
            Job Description: {jd_text[:4000]}
            Return only the complete tailored resume.
            """

            with st.spinner("Crafting your tailored resume..."):
                response = llm.invoke(full_prompt)
                tailored = response.content

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(tailored)
            st.markdown('</div>', unsafe_allow_html=True)

            pdf_bytes = create_tailored_pdf(tailored)

            st.download_button(
                label="↓ Download Optimized Resume PDF",
                data=pdf_bytes,
                file_name="ResumeFlow_Optimized_Resume.pdf",
                mime="application/pdf",
                use_container_width=True
            )

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem 1rem;">
        <h3>Ready to level up your resume?</h3>
        <p>Upload your current resume and the job description you want to target.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#777; font-size:0.9rem;'>"
    "ResumeFlow • Minimal AI Resume Optimizer • Made for ambitious students"
    "</p>", 
    unsafe_allow_html=True
)
