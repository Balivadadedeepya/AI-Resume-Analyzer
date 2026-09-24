# 🤖 AI Resume Analyzer

An AI-powered web application that analyzes a resume against a target job description using Natural Language Processing (NLP) and Machine Learning techniques.

🔗 **Live Demo:** https://ai-resume-analyzer-gcf3hbx27wl62vrbo7bkux.streamlit.app/

---

## 🚀 Features

- 📄 Upload resume in PDF format
- 🎯 Compare resume with a target job description
- 🧠 NLP-based text similarity analysis
- 🛠️ Automatic skill extraction
- ✅ Matching skill detection
- ❌ Missing skill detection
- 📊 Skill coverage analysis
- 📈 Visual skill comparison
- 💡 Personalized resume improvement recommendations

---

## 🧠 How It Works

1. Upload your resume as a PDF.
2. Paste the target job description.
3. The application extracts text from the resume.
4. Relevant technical skills are detected.
5. TF-IDF and cosine similarity are used to compare the resume with the job description.
6. Matching and missing skills are identified.
7. The application provides improvement recommendations.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF
- Cosine Similarity
- PyPDF2
- Regular Expressions

---

## 📊 Analysis Output

The application provides:

- Resume–Job Similarity Score
- Required Skill Coverage
- Matching Skills
- Missing Skills
- Detected Resume Skills
- Skill Coverage Visualization
- Resume Improvement Recommendations

---

## 📁 Project Structure

```text
AI-Resume-Analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
