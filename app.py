import io
import re

import PyPDF2
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SKILL DATABASE
# ============================================================

SKILLS = [
    "python",
    "java",
    "c++",
    "javascript",
    "typescript",
    "sql",
    "html",
    "css",
    "react",
    "node.js",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "natural language processing",
    "nlp",
    "computer vision",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "opencv",
    "matplotlib",
    "seaborn",
    "power bi",
    "tableau",
    "excel",
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "google cloud",
    "fastapi",
    "flask",
    "streamlit",
    "data science",
    "data analysis",
    "data analytics",
    "statistics",
    "rest api",
    "mongodb",
    "mysql",
    "postgresql"
]


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF resume."""

    pdf_bytes = uploaded_file.read()
    pdf_file = io.BytesIO(pdf_bytes)

    reader = PyPDF2.PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Clean text before NLP processing."""

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"[^a-zA-Z0-9+#.\- ]",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):
    """Detect technical skills from text."""

    text = clean_text(text)

    detected_skills = []

    for skill in SKILLS:

        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):
            detected_skills.append(skill)

    return sorted(set(detected_skills))


# ============================================================
# TF-IDF SIMILARITY
# ============================================================

def calculate_similarity(resume_text, job_description):
    """Calculate similarity between resume and job description."""

    documents = [
        clean_text(resume_text),
        clean_text(job_description)
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return similarity * 100


# ============================================================
# SKILL COMPARISON
# ============================================================

def compare_skills(resume_skills, job_skills):

    resume_set = set(resume_skills)
    job_set = set(job_skills)

    matching_skills = sorted(
        resume_set.intersection(job_set)
    )

    missing_skills = sorted(
        job_set - resume_set
    )

    return matching_skills, missing_skills


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    resume_skills,
    job_skills,
    matching_skills,
    missing_skills,
    similarity
):

    recommendations = []

    if missing_skills:

        recommendations.append(
            "Consider adding relevant missing skills "
            "only if you genuinely have experience with them."
        )

    if similarity < 40:

        recommendations.append(
            "The resume has relatively low textual similarity "
            "to the target job description. Consider tailoring "
            "your project and experience descriptions."
        )

    elif similarity >= 70:

        recommendations.append(
            "The resume has substantial textual overlap "
            "with the target job description."
        )

    else:

        recommendations.append(
            "The resume has moderate textual similarity "
            "with the target job description."
        )

    if len(resume_skills) < 5:

        recommendations.append(
            "Consider clearly listing your technical skills "
            "in a dedicated Skills section."
        )

    if "github" not in resume_skills:

        recommendations.append(
            "Consider adding your GitHub profile if you "
            "have relevant public projects."
        )

    return recommendations


# ============================================================
# HEADER
# ============================================================

st.title("🤖 AI Resume Analyzer")

st.write(
    "Analyze your resume against a target job description "
    "using Natural Language Processing and Machine Learning."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

left_column, right_column = st.columns(2)


with left_column:

    st.subheader("📄 Upload Resume")

    uploaded_resume = st.file_uploader(
        "Upload your resume in PDF format",
        type=["pdf"]
    )


with right_column:

    st.subheader("🎯 Target Job Description")

    job_description = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder="Paste the complete job description..."
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🚀 Analyze Resume",
    use_container_width=True
)


if analyze:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if uploaded_resume is None:

        st.error(
            "Please upload your resume first."
        )

    elif not job_description.strip():

        st.error(
            "Please enter a job description."
        )

    else:

        with st.spinner(
            "Analyzing your resume..."
        ):

            # Extract resume text
            resume_text = extract_text_from_pdf(
                uploaded_resume
            )

            if not resume_text.strip():

                st.error(
                    "Could not extract text from this PDF. "
                    "Please try a text-based PDF."
                )

                st.stop()

            # Extract skills
            resume_skills = extract_skills(
                resume_text
            )

            job_skills = extract_skills(
                job_description
            )

            # Calculate similarity
            similarity = calculate_similarity(
                resume_text,
                job_description
            )

            # Compare skills
            matching_skills, missing_skills = compare_skills(
                resume_skills,
                job_skills
            )

            # Generate recommendations
            recommendations = generate_recommendations(
                resume_skills,
                job_skills,
                matching_skills,
                missing_skills,
                similarity
            )


        st.success(
            "Analysis completed successfully!"
        )

        st.divider()


        # ====================================================
        # SCORE SECTION
        # ====================================================

        st.subheader("📊 Resume Analysis")


        score_column, skill_column = st.columns(2)


        with score_column:

            st.metric(
                "Resume–Job Text Similarity",
                f"{similarity:.1f}%"
            )

            st.progress(
                min(int(similarity), 100)
            )


        with skill_column:

            if job_skills:

                skill_match_percentage = (
                    len(matching_skills)
                    / len(job_skills)
                ) * 100

            else:

                skill_match_percentage = 0


            st.metric(
                "Required Skill Coverage",
                f"{skill_match_percentage:.1f}%"
            )

            st.progress(
                min(
                    int(skill_match_percentage),
                    100
                )
            )


        st.divider()


        # ====================================================
        # MATCHING / MISSING SKILLS
        # ====================================================

        matching_column, missing_column = st.columns(2)


        with matching_column:

            st.subheader(
                "✅ Matching Skills"
            )

            if matching_skills:

                for skill in matching_skills:

                    st.write(
                        f"✅ {skill.title()}"
                    )

            else:

                st.info(
                    "No matching skills were detected."
                )


        with missing_column:

            st.subheader(
                "❌ Missing / Undetected Skills"
            )

            if missing_skills:

                for skill in missing_skills:

                    st.write(
                        f"❌ {skill.title()}"
                    )

            else:

                st.success(
                    "No missing skills were detected "
                    "from the current skill database."
                )


        st.divider()


        # ====================================================
        # DETECTED RESUME SKILLS
        # ====================================================

        st.subheader(
            "🧠 Skills Detected in Your Resume"
        )


        if resume_skills:

            st.write(
                ", ".join(
                    skill.title()
                    for skill in resume_skills
                )
            )

        else:

            st.info(
                "No skills from the current skill database "
                "were detected."
            )


        st.divider()


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.subheader(
            "💡 Recommendations"
        )


        for recommendation in recommendations:

            st.write(
                f"• {recommendation}"
            )


        st.divider()


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.caption(
            "Note: This tool provides text similarity and "
            "skill-matching analysis. It is not a hiring "
            "decision system and does not guarantee job "
            "eligibility or interview selection."
        )
