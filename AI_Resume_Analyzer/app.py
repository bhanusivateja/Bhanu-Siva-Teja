import streamlit as st
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# ---------------- PAGE SETTINGS ---------------- #

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)


# ---------------- AI BACKGROUND ---------------- #

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1677442135703-1787eea5ce01");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}

[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}

.main {
background-color: rgba(255,255,255,0.9);
padding: 25px;
border-radius: 10px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


# ---------------- INTRO SECTION ---------------- #

st.title("🤖 AI Resume Analyzer")

st.markdown("""
### Intelligent Resume Screening using Artificial Intelligence

This web application demonstrates how **Artificial Intelligence and Natural Language Processing (NLP)**  
can be used to analyze resumes and compare them with job descriptions.

The system calculates a **resume match score**, extracts **skills**, and provides **suggestions to improve the resume**.

---

### 🎯 Project Objective

The goal of this project is to simulate an **Applicant Tracking System (ATS)** used by companies to filter resumes.

It helps job seekers understand:

• How well their resume matches a job description  
• Which skills are already matched  
• Which skills are missing  

---

### 🛠 Technologies Used

• Python  
• Streamlit  
• Scikit-learn  
• Natural Language Processing  
• PDFMiner  
• Matplotlib  

---

### 👨‍💻 Developed By

**Bhanu Siva Teja**  
B.Tech – CSE (Artificial Intelligence & Machine Learning)

This project demonstrates **AI-based resume analysis and intelligent job matching systems used in modern recruitment platforms.**
""")

st.write("---")


# ---------------- SIDEBAR ---------------- #

st.sidebar.title("AI Resume Analyzer")

st.sidebar.info("""
Upload your resume and paste the job description to analyze how well your resume matches the job requirements.

This tool simulates a basic **ATS Resume Screening System** used by recruiters.
""")


# ---------------- FUNCTIONS ---------------- #

def extract_resume_text(file):
    text = extract_text(file)
    return text


def calculate_match(resume_text, job_desc):

    documents = [resume_text, job_desc]

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)

    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return score[0][0] * 100


skills_list = [
    "python","java","c++","machine learning",
    "deep learning","data analysis","sql",
    "tensorflow","pandas","numpy","aws",
    "html","css","javascript"
]


def extract_skills(text):

    text = text.lower()
    found_skills = []

    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills


# ---------------- MAIN APP ---------------- #

st.header("📄 Upload Resume and Job Description")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

job_desc = st.text_area("Enter Job Description")


# ---------------- ANALYSIS ---------------- #

if resume_file and job_desc:

    resume_text = extract_resume_text(resume_file)

    match_score = calculate_match(resume_text, job_desc)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc)

    matched_skills = list(set(resume_skills) & set(job_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))


    # Resume Match Score
    st.subheader("📊 Resume Match Score")

    st.progress(int(match_score))

    st.success(f"{round(match_score,2)} % Match")


    # ATS Score
    ats_score = int(match_score)

    st.subheader("📈 ATS Resume Score")

    st.write(ats_score, "/ 100")


    # Matched Skills
    st.subheader("✅ Matched Skills")

    if matched_skills:
        for skill in matched_skills:
            st.write("✔", skill)
    else:
        st.write("No matched skills found")


    # Missing Skills
    st.subheader("❌ Missing Skills")

    if missing_skills:
        for skill in missing_skills:
            st.write("❌", skill)
    else:
        st.write("No missing skills")


    # Suggestions
    st.subheader("💡 Suggestions to Improve Resume")

    if missing_skills:

        st.write("Consider adding these skills:")

        for skill in missing_skills:
            st.write("➜", skill)

    else:

        st.write("Your resume already matches well!")


    # Graph
    labels = ['Matched Skills','Missing Skills']
    values = [len(matched_skills), len(missing_skills)]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    st.subheader("📊 Skill Analysis Graph")

    st.pyplot(fig)