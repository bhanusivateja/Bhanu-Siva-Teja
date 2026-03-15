import streamlit as st
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# Function to extract text from PDF
def extract_resume_text(file):
    text = extract_text(file)
    return text


# Function to calculate similarity score
def calculate_match(resume_text, job_desc):

    documents = [resume_text, job_desc]

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)

    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return score[0][0] * 100


# Skills list
skills_list = [
    "python","java","c++","machine learning",
    "deep learning","data analysis","sql",
    "tensorflow","pandas","numpy","aws",
    "html","css","javascript"
]


# Extract skills
def extract_skills(text):

    text = text.lower()
    found_skills = []

    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills


# Streamlit UI
st.title("AI Resume Analyzer")

st.write("Upload your resume and compare it with job description")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

job_desc = st.text_area("Enter Job Description")


if resume_file and job_desc:

    resume_text = extract_resume_text(resume_file)

    match_score = calculate_match(resume_text, job_desc)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc)

    matched_skills = list(set(resume_skills) & set(job_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))

    # Resume match score
    st.subheader("Resume Match Score")
    st.success(str(round(match_score,2)) + "% Match")

    # ATS Score
    ats_score = int(match_score)

    st.subheader("ATS Resume Score")
    st.write(ats_score, "/ 100")

    # Matched Skills
    st.subheader("Matched Skills")

    if matched_skills:
        for skill in matched_skills:
            st.write("✔", skill)
    else:
        st.write("No matched skills found")

    # Missing Skills
    st.subheader("Missing Skills")

    if missing_skills:
        for skill in missing_skills:
            st.write("❌", skill)
    else:
        st.write("No missing skills")

    # Suggestions
    st.subheader("Suggestions to Improve Resume")

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

    st.subheader("Skill Analysis Graph")
    st.pyplot(fig)