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

load_dotenv()

st.set_page_config(page_title="AI Resume Optimizer", page_icon="📄", layout="wide")
st.title("🚀 AI Resume Optimizer & Job Matcher")
st.markdown("Upload your resume + paste Job Description → Get score, improvements & tailored resume")

# Load Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load sample jobs
@st.cache_data
def load_jobs():
    return pd.read_csv("jobs.csv")

jobs_df = load_jobs()

# Function to extract text from PDF
def extract_resume_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# Function to create tailored PDF
def create_tailored_pdf(resume_text, tailored_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "TAILORED RESUME")
    y -= 40
    c.setFont("Helvetica", 12)
    for line in tailored_text.split("\n"):
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line)
        y -= 15
    c.save()
    buffer.seek(0)
    return buffer

# Upload resume
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

# Paste Job Description
jd_text = st.text_area("Paste the Job Description here", height=200)

if uploaded_file and jd_text:
    with st.spinner("Analyzing your resume..."):
        resume_text = extract_resume_text(uploaded_file)
        
        # Create embeddings
        resume_embedding = embeddings.embed_query(resume_text)
        jd_embedding = embeddings.embed_query(jd_text)
        
        # Cosine similarity score
        similarity = cos_sim(torch.tensor([resume_embedding]), torch.tensor([jd_embedding]))[0][0].item()
        match_score = round(similarity * 100, 1)
        
        st.subheader(f"📊 Match Score: **{match_score}%**")
        
        # Show sample job matches from CSV (bonus)
        st.caption("Also matches these sample jobs (from our database):")
        job_embeddings = [embeddings.embed_query(desc) for desc in jobs_df['description']]
        similarities = [cos_sim(torch.tensor([resume_embedding]), torch.tensor([je]))[0][0].item() for je in job_embeddings]
        jobs_df['score'] = [round(s*100, 1) for s in similarities]
        top_jobs = jobs_df.nlargest(3, 'score')[['title', 'score']]
        st.dataframe(top_jobs, hide_index=True)
        
        # LLM Analysis & Suggestions
        prompt = f"""
        You are an expert Indian career coach for campus placements.
        Resume: {resume_text[:4000]}
        Job Description: {jd_text[:4000]}
        
        Tasks:
        1. Give match percentage (already calculated: {match_score}%)
        2. List 5-7 keywords/skills from JD that are MISSING in resume.
        3. Suggest 4-5 specific bullet point improvements.
        4. Rewrite the entire resume tailored to this JD (keep original structure but make it stronger).
        
        Return in this format:
        MISSING SKILLS: ...
        SUGGESTIONS: ...
        TAILORED RESUME: ...
        """
        
        response = llm.invoke(prompt)
        result = response.content
        
        st.subheader("✅ AI Analysis & Tailored Resume")
        st.markdown(result)
        
        # Extract tailored resume part
        if "TAILORED RESUME:" in result:
            tailored_part = result.split("TAILORED RESUME:")[-1].strip()
        else:
            tailored_part = result
        
        # Download button
        pdf_bytes = create_tailored_pdf(resume_text, tailored_part)
        st.download_button(
            label="📥 Download Tailored Resume as PDF",
            data=pdf_bytes,
            file_name="Tailored_Resume.pdf",
            mime="application/pdf"
        )
        
        st.success("✅ Done! You can now copy the suggestions and use the downloaded PDF.")

st.caption("Built with LangChain + Groq + FAISS-style embeddings | Perfect for Uttar Pradesh college placements")
