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
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            rgba(49, 51, 63, 0.08),
            rgba(120, 120, 120, 0.05)
        );
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.3rem;
        font-size: 2.6rem;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.8;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        text-align: center;
        min-height: 130px;
    }

    .metric-title {
        font-size: 0.9rem;
        opacity: 0.7;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    .skill-box {
        padding: 0.65rem 0.9rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 0.5rem;
    }

    .footer {
        text-align: center;
        opacity: 0.65;
        padding: 2rem 0 1rem 0;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SKILL DATABASE
# ============================================================

SKILL_PATTERNS = {
    "Python": [r"\bpython\b"],
    "Java": [r"\bjava\b"],
    "C++": [r"\bc\+\+\b"],
    "JavaScript": [r"\bjavascript\b"],
    "TypeScript": [r"\btypescript\b"],
    "SQL": [r"\bsql\b"],
    "HTML": [r"\bhtml\b"],
    "CSS": [r"\bcss\b"],

    "React": [r"\breact\b", r"\breact\.js\b"],
    "Node.js": [r"\bnode\.js\b", r"\bnodejs\b"],
    "Flask": [r"\bflask\b"],
    "FastAPI": [r"\bfastapi\b"],
    "Streamlit": [r"\bstreamlit\b"],

    "Machine Learning": [
        r"\bmachine learning\b",
        r"\bmachine-learning\b",
        r"\bml\b",
    ],
    "Deep Learning": [
        r"\bdeep learning\b",
        r"\bdeep-learning\b",
        r"\bdl\b",
    ],
    "Artificial Intelligence": [
        r"\bartificial intelligence\b",
        r"\bai\b",
    ],
    "NLP": [
        r"\bnatural language processing\b",
        r"\bnlp\b",
    ],
    "Computer Vision": [
        r"\bcomputer vision\b",
        r"\bopencv\b",
    ],

    "TensorFlow": [r"\btensorflow\b"],
    "PyTorch": [r"\bpytorch\b"],
    "Scikit-learn": [
        r"\bscikit-learn\b",
        r"\bscikit learn\b",
        r"\bsklearn\b",
    ],

    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b", r"\bnumPy\b"],
    "Matplotlib": [r"\bmatplotlib\b"],
    "Seaborn": [r"\bseaborn\b"],

    "Data Science": [r"\bdata science\b"],
    "Data Analysis": [r"\bdata analysis\b"],
    "Data Analytics": [r"\bdata analytics\b"],
    "Statistics": [r"\bstatistics\b"],
    "Excel": [r"\bexcel\b"],
    "Power BI": [r"\bpower bi\b"],
    "Tableau": [r"\btableau\b"],

    "Git": [r"\bgit\b"],
    "GitHub": [r"\bgithub\b"],
    "Docker": [r"\bdocker\b"],

    "AWS": [r"\baws\b", r"\bamazon web services\b"],
    "Azure": [r"\bazure\b"],
    "Google Cloud": [
        r"\bgoogle cloud\b",
        r"\bgcp\b",
    ],

    "REST API": [
        r"\brest api\b",
        r"\brestful api\b",
        r"\brestful\b",
    ],
    "MongoDB": [r"\bmongodb\b"],
    "MySQL": [r"\bmysql\b"],
    "PostgreSQL": [r"\bpostgresql\b", r"\bpostgres\b"],
}


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_bytes = uploaded_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))

        pages = []

        for page in pdf_reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    except Exception as error:
        st.error(f"Could not read the PDF: {error}")
        return ""


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):
    cleaned = clean_text(text)
    detected = []

    for skill, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                detected.append(skill)
                break

    return sorted(detected)


# ============================================================
# TEXT SIMILARITY
# ============================================================

def calculate_similarity(resume_text, job_text):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)

    if not resume_clean or not job_clean:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        vectors = vectorizer.fit_transform(
            [resume_clean, job_clean]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return float(similarity * 100)

    except Exception:
        return 0.0


# ============================================================
# SKILL COMPARISON
# ============================================================

def compare_skills(resume_skills, job_skills):
    resume_set = set(resume_skills)
    job_set = set(job_skills)

    matching = sorted(resume_set.intersection(job_set))
    missing = sorted(job_set - resume_set)

    if job_set:
        coverage = len(matching) / len(job_set) * 100
    else:
        coverage = 0.0

    return matching, missing, coverage


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    similarity,
    coverage,
    matching,
    missing,
    resume_skills
):
    recommendations = []

    if similarity < 40:
        recommendations.append(
            "Your resume has relatively low textual similarity "
            "with the target role. Consider tailoring your project "
            "and experience descriptions to the job description."
        )

    if coverage < 50 and missing:
        recommendations.append(
            "Consider adding relevant missing skills only if "
            "you genuinely have experience with them."
        )

    if not matching:
        recommendations.append(
            "Very few required skills were detected in both texts. "
            "Review the job requirements and highlight relevant "
            "experience in your resume."
        )

    if len(resume_skills) < 8:
        recommendations.append(
            "Your resume contains a limited number of detectable "
            "technical skills. Consider clearly listing relevant "
            "technologies used in your projects."
        )

    if matching:
        recommendations.append(
            "Keep your strongest matching skills visible in your "
            "Summary, Skills, and Project sections."
        )

    recommendations.append(
        "Use measurable achievements where possible, such as "
        "accuracy, performance improvements, project scale, "
        "or time saved."
    )

    return recommendations


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📄 AI Resume Analyzer</h1>
        <p>
            Analyze your resume against a target job description
            using Natural Language Processing and Machine Learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT SECTION
# ============================================================

left, right = st.columns(2)

with left:
    st.subheader("📄 Upload Resume")

    uploaded_file = st.file_uploader(
        "Upload your resume in PDF format",
        type=["pdf"],
        help="Upload a text-based PDF resume.",
    )

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

with right:
    st.subheader("🎯 Target Job Description")

    job_description = st.text_area(
        "Paste the job description here",
        placeholder=(
            "Paste the complete job description...\n\n"
            "Example: Python, Machine Learning, SQL, "
            "NLP, TensorFlow..."
        ),
        height=220,
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("")

analyze = st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not uploaded_file:
        st.warning("Please upload a resume PDF first.")

    elif not job_description.strip():
        st.warning("Please enter a target job description.")

    else:

        with st.spinner("Analyzing your resume..."):

            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text.strip():
                st.error(
                    "No readable text was found in the PDF. "
                    "Please upload a text-based PDF."
                )
                st.stop()

            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_description)

            similarity = calculate_similarity(
                resume_text,
                job_description
            )

            matching, missing, coverage = compare_skills(
                resume_skills,
                job_skills
            )

            recommendations = generate_recommendations(
                similarity,
                coverage,
                matching,
                missing,
                resume_skills
            )

        st.success("✅ Analysis completed successfully!")

        st.divider()

        # ====================================================
        # METRICS
        # ====================================================

        st.header("📊 Resume Analysis")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        Resume–Job Similarity
                    </div>
                    <div class="metric-value">
                        {similarity:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        Required Skill Coverage
                    </div>
                    <div class="metric-value">
                        {coverage:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        Skills Detected
                    </div>
                    <div class="metric-value">
                        {len(resume_skills)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Progress indicators

        st.write("**Resume–Job Similarity**")
        st.progress(
            min(max(similarity / 100, 0.0), 1.0)
        )

        st.write("**Required Skill Coverage**")
        st.progress(
            min(max(coverage / 100, 0.0), 1.0)
        )

        st.divider()

        # ====================================================
        # SKILL COMPARISON
        # ====================================================

        st.header("🎯 Skill Match")

        skill_left, skill_right = st.columns(2)

        with skill_left:
            st.subheader("✅ Matching Skills")

            if matching:
                for skill in matching:
                    st.markdown(
                        f'<div class="skill-box">✅ {skill}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "No matching skills were detected."
                )

        with skill_right:
            st.subheader("❌ Missing / Undetected Skills")

            if missing:
                for skill in missing:
                    st.markdown(
                        f'<div class="skill-box">❌ {skill}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success(
                    "All detected job skills are present!"
                )

        st.divider()

        # ====================================================
        # SKILLS DETECTED
        # ====================================================

        st.header("🧠 Skills Detected in Your Resume")

        if resume_skills:

            skill_columns = st.columns(3)

            for index, skill in enumerate(resume_skills):
                with skill_columns[index % 3]:
                    st.markdown(
                        f'<div class="skill-box">🔹 {skill}</div>',
                        unsafe_allow_html=True,
                    )

        else:
            st.info(
                "No skills from the current skill database "
                "were detected."
            )

        # ====================================================
        # SKILL CHART
        # ====================================================

        if matching or missing:

            st.divider()

            st.header("📈 Skill Coverage Overview")

            chart_data = {
                "Category": [
                    "Matching Skills",
                    "Missing Skills"
                ],
                "Count": [
                    len(matching),
                    len(missing)
                ],
            }

            st.bar_chart(
                chart_data,
                x="Category",
                y="Count",
            )

        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.divider()

        st.header("💡 Recommendations")

        for recommendation in recommendations:
            st.markdown(
                f"• {recommendation}"
            )

        # ====================================================
        # TEXT PREVIEW
        # ====================================================

        st.divider()

        with st.expander("🔍 View extracted resume text"):

            preview = resume_text[:5000]

            st.text(
                preview
                + (
                    "\n\n[Preview truncated...]"
                    if len(resume_text) > 5000
                    else ""
                )
            )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.info(
            "ℹ️ This tool is intended for resume improvement "
            "and educational purposes. It is not a hiring "
            "decision system and should not be used as the sole "
            "basis for employment decisions."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with Python • Streamlit • NLP • Machine Learning
        <br>
        AI Resume Analyzer
    </div>
    """,
    unsafe_allow_html=True,
)
