import streamlit as st

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Analyzer")
st.write(
    "Analyze your resume against a job description using "
    "Natural Language Processing and Machine Learning."
)

st.divider()

# Resume upload
st.subheader("📄 Upload Your Resume")

resume = st.file_uploader(
    "Upload your resume",
    type=["pdf"],
    help="Upload your resume in PDF format."
)

st.subheader("🎯 Target Job Description")

job_description = st.text_area(
    "Paste the job description here",
    height=250,
    placeholder="Paste the complete job description here..."
)

if st.button("🔍 Analyze Resume", type="primary"):

    if resume is None:
        st.warning("Please upload your resume first.")

    elif not job_description.strip():
        st.warning("Please enter a job description.")

    else:
        st.success("Resume and job description received!")

        st.info(
            "The AI analysis engine will be connected in the next stage."
        )

st.divider()

st.caption("AI Resume Analyzer | Built with Python, NLP & Machine Learning")
