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

st.set_page_config(
    page_title="AI Resume Optimizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS
st.markdown("""
    <style>
    .main {background-color: #0e1117; color: #ffffff;}
    .stButton>button {
        width: 100%; 
        height: 3.2rem; 
        background: linear-gradient(90deg, #00c853, #64dd17);
        color: white; 
        border: none; 
        border-radius: 8px;
        font-weight: bold;
    }
    .match-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a40);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #00c853;
    }
    .score {font-size: 3.5rem; font-weight: bold; margin: 10px 0;}
    .high {color: #00ff88;}
    .medium {color: #ffcc00;}
    .low {color: #ff5252;}
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Optimizer")
st.markdown("**Get AI-powered resume analysis, skill matching & tailored resume in seconds**")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/resume.png", width=80)
    st.header("How it Works")
    st.markdown("""
    1. Upload your current resume (PDF)
    2. Paste the target Job Description
    3. Click **Analyze**
    4. Get score + suggestions + tailored resume
    """)
    st.divider()
    st.success("✅ Optimized for Campus Placements 2026")
    st.caption("Built with Groq • LangChain • Streamlit")

# API Key Setup
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud → Secrets")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data
def load_jobs():
    try:
        return pd.read_csv("jobs.csv")
    except:
        return pd.DataFrame({
            "title": ["Software Engineer", "Data Analyst", "ML Engineer"],
            "description": ["Python, Django, SQL, AWS", "Python, SQL, Power BI", "Python, PyTorch, NLP"]
        })

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
    y = letter[1] - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "TAILORED RESUME - AI OPTIMIZED")
    y -= 40
    c.setFont("Helvetica", 11)
    for line in tailored_text.split("\n"):
        if y < 60:
            c.showPage()
            y = letter[1] - 60
        c.drawString(50, y, line.strip())
        y -= 14
    c.save()
    buffer.seek(0)
    return buffer

# Main Layout
col1, col2 = st.columns([1, 1.2])

with col1:
    uploaded_file = st.file_uploader("📤 Upload Your Resume (PDF)", type="pdf")

with col2:
    jd_text = st.text_area("📋 Paste Job Description", height=220, 
                          placeholder="Copy and paste the full job description here...")

if uploaded_file and jd_text:
    if st.button("🔍 Analyze Resume & Generate Tailored Version", type="primary"):
        
        with st.spinner("AI is analyzing your resume... This may take 10-15 seconds"):
            resume_text = extract_resume_text(uploaded_file)

            # Calculate Match Score
            resume_emb = embeddings.embed_query(resume_text)
            jd_emb = embeddings.embed_query(jd_text)
            similarity = cos_sim(torch.tensor([resume_emb]), torch.tensor([jd_emb]))[0][0].item()
            match_score = round(similarity * 100, 1)

            # Score Styling
            if match_score >= 75:
                score_class = "high"
                emoji = "🎯 Excellent Match!"
            elif match_score >= 55:
                score_class = "medium"
                emoji = "👍 Good Match"
            else:
                score_class = "low"
                emoji = "⚠️ Needs Improvement"

            # Display Results in Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Match Score", "💡 Suggestions", "📝 Tailored Resume"])

            with tab1:
                st.markdown(f"""
                <div class="match-card">
                    <h3>{emoji}</h3>
                    <div class="score {score_class}">{match_score}%</div>
                    <p>Resume-Job Match Score</p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("Top Matching Sample Roles")
                job_embs = [embeddings.embed_query(desc) for desc in jobs_df['description']]
                sims = [cos_sim(torch.tensor([resume_emb]), torch.tensor([je]))[0][0].item() for je in job_embs]
                jobs_df['score'] = [round(s*100, 1) for s in sims]
                st.dataframe(jobs_df.nlargest(3, 'score')[['title', 'score']], hide_index=True, use_container_width=True)

            with tab2:
                prompt = f"""
                You are an expert career coach for Indian engineering students.
                Resume: {resume_text[:4500]}
                Job Description: {jd_text[:4500]}

                Return strictly in this format:
                MISSING SKILLS:\n- skill1\n- skill2...
                IMPROVEMENT SUGGESTIONS:\n1. ...\n2. ...
                """

                response = llm.invoke(prompt)
                st.markdown(response.content)

            with tab3:
                full_prompt = f"""
                You are an expert resume writer. Rewrite the following resume to perfectly match the job description.
                Keep it professional, use strong action verbs, and quantify where possible.

                Original Resume:
                {resume_text[:4000]}

                Job Description:
                {jd_text[:4000]}

                Return only the full tailored resume in clean bullet point format.
                """

                with st.spinner("Generating tailored resume..."):
                    tailored_response = llm.invoke(full_prompt)
                    tailored_text = tailored_response.content

                st.markdown("### Your AI-Tailored Resume")
                st.markdown(tailored_text)

                pdf_bytes = create_tailored_pdf(tailored_text)

                st.download_button(
                    label="📥 Download Tailored Resume as PDF",
                    data=pdf_bytes,
                    file_name="AI_Tailored_Resume.pdf",
                    mime="application/pdf",
                    type="primary"
                )

else:
    st.info("👆 Please upload your resume and paste the Job Description to begin analysis.")

st.divider()
st.markdown("Made with ❤️ for students in Uttar Pradesh | Groq • LangChain • Streamlit")
