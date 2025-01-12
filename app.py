import os
import re
import PyPDF2 as pdf
import streamlit as st
from functools import partial
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from typing import Annotated, TypedDict
import google.generativeai as genai
import io
import time
import plotly.express as px
import pandas as pd
st.set_page_config(layout="wide")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define LangGraph State
class State(TypedDict):
    resume_text: str
    job_description: str
    scores: Annotated[list, "Add messages"]
    individual_scores: dict

def skill_analysis(state, resume_text):
    prompt = f"""
    You are a highly skilled resume evaluator tasked with analyzing the relevance and quality of skills listed in a resume against a job description (JD).
    Evaluate the skills based on the following criteria, and provide a detailed breakdown of scores, examples, and actionable feedback:

    **Skills Match Evaluation**
    1. Relevance to JD: [Score]%
        - **Matched Skills**: List all explicitly mentioned skills in the resume that match the JD, including specific technologies, frameworks, and methodologies (one line).
        - **Missing Skills**: Identify essential skills, technologies, or qualifications mentioned in the JD that are absent or underrepresented in the resume (one line).
    2. Specificity and Depth: [Score]%
        - **Description**: Evaluate how specifically and thoroughly the skills are described. Include details like experience level, proficiency, and usage in context (e.g., "Advanced Python programming with 5 years of experience") (one line).
    3. Relevance to Role and Industry: [Score]%
        - **Description**: Assess how well the listed skills align with the role and industry requirements outlined in the JD, with a focus on industry-standard tools and methodologies (one line).
    4. Level of Expertise: [Rating]/5
        - **Description**: Rate the proficiency and expertise demonstrated in each skill listed on the resume (e.g., beginner, intermediate, advanced). Provide reasoning based on the description in the resume (one line).
    5. Overall Skills Match Score: [Score]% 
        - **Summary**: Provide a concise overview of the resume’s skills match against the JD, noting strengths and areas for improvement (one line).
    6. Actionable Feedback: Suggest ways to improve the alignment of the skills section with the JD, including recommendations for emphasizing relevant skills or gaining missing skills (one line).

    Resume: {resume_text}
    Job Description: {state['job_description']}
    """
    try:
        response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
        breakdown = response.text.strip()
        state.setdefault("individual_scores", {})["Skills Match"] = breakdown
    except Exception as e:
        state.setdefault("individual_scores", {})["Skills Match"] = f"Error: {str(e)}"

def project_analysis(state, resume_text):
    prompt = f"""
    You are an experienced resume evaluator specializing in assessing the relevance and quality of projects listed in a resume against a job description (JD).
    Evaluate the match based on the following criteria, providing a detailed breakdown with scores, examples, and improvement suggestions:

    **Project Analysis Evaluation**
    1. Relevance to JD: [Score]%
        - **Matched Projects**: List the projects from the resume that are directly relevant to the JD requirements, emphasizing key aspects that match the role (one line).
        - **Missing Projects**: Identify project types or experiences that are mentioned in the JD but are missing or underrepresented in the resume (one line).
    2. Coverage of Essential Project Experience: [Score]%
        - **Description**: Assess how well the listed projects cover the critical requirements and responsibilities of the JD. Mention any key projects that should be highlighted in the resume to better align with the JD (one line).
    3. Specificity and Detail: [Score]%
        - **Description**: Evaluate the level of detail provided about the projects, including outcomes, measurable results, and relevant technologies used (one line).
    4. Level of Expertise: [Rating]/5
        - **Description**: Rate the expertise demonstrated in the projects. Consider the complexity, scale, and impact of each project described in the resume (e.g., beginner, intermediate, advanced) (one line).
    5. Overall Project Match Score: [Score]%
        - **Summary**: Summarize the overall match between the resume’s listed projects and the JD, highlighting key strengths and areas for improvement (one line).
    6. Feedback for Improvement: Provide actionable suggestions for improving the project descriptions, including additional relevant projects, better articulation of impact, and quantifiable results (one line).

    Resume: {resume_text}
    Job Description: {state['job_description']}
    """
    try:
        response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
        breakdown = response.text.strip()
        state.setdefault("individual_scores", {})["Project Analysis"] = breakdown
    except Exception as e:
        state.setdefault("individual_scores", {})["Project Analysis"] = f"Error: {str(e)}"

def experience_analysis(state, resume_text):
    prompt = f"""
    You are a professional resume evaluator tasked with analyzing the experience section of a resume against a job description (JD).
    Evaluate the match based on the following criteria, providing detailed feedback with scores, examples, and actionable suggestions for improvement:

    **Experience Analysis Evaluation**
    1. Relevance to JD: [Score]%
        - **Matched Experience**: Highlight specific experiences that directly align with the JD, showcasing how the candidate's past roles meet the JD’s requirements (one line).
        - **Missing Experience**: Identify critical experiences mentioned in the JD that are absent from the resume or inadequately covered (one line).
    2. Coverage of Essential Experience: [Score]%
        - **Description**: Evaluate how well the experience section covers the critical aspects of the JD, including job responsibilities, leadership roles, and skills (one line).
    3. Specificity and Impact: [Score]%
        - **Description**: Assess the specificity of the descriptions of each role, including measurable achievements and contributions (e.g., “Increased sales by 20%” or “Managed a team of 10 developers”) (one line).
    4. Level of Expertise: [Rating]/5
        - **Description**: Rate the expertise demonstrated in the experience section, considering the complexity of tasks performed and the level of responsibility (e.g., beginner, intermediate, advanced) (one line).
    5. Overall Experience Match Score: [Score]%
        - **Summary**: Provide an overall evaluation of how well the experience section aligns with the JD, noting strengths and areas for improvement (one line).
    6. Actionable Feedback: Offer suggestions on how to improve the experience section, such as emphasizing relevant experiences or improving the clarity of job descriptions (one line).

    Resume: {resume_text}
    Job Description: {state['job_description']}
    """
    try:
        response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
        breakdown = response.text.strip()
        state.setdefault("individual_scores", {})["Experience Analysis"] = breakdown
    except Exception as e:
        state.setdefault("individual_scores", {})["Experience Analysis"] = f"Error: {str(e)}"

def rewrite_suggestions(state, resume_text):
    prompt = f"""
    You are a highly skilled resume writer and editor specializing in crafting impactful and professional resumes. 
    Your task is to evaluate the provided resume text and suggest improvements for clarity, grammar, alignment with the job description, and overall effectiveness. Focus on enhancing poorly explained sections, ensuring relevance to the target job, and optimizing for readability and professional presentation.

    ### Key Guidelines:
    1. **Education Section**:
        - Highlight the most relevant academic qualifications for the job.
        - Ensure reverse chronological order, including details like institution name, degree type, major/minor, graduation year, and special recognitions (e.g., honors, scholarships).
        - Include online courses or certifications only if they are relevant to the job description.

    2. **Experience Section**:
        - Emphasize specific achievements and quantifiable results (e.g., "Increased sales by 25% in Q4").
        - Clearly describe roles and responsibilities using action verbs and avoiding vague terms.
        - Align experiences with skills and competencies highlighted in the job description.

    3. **Skills Section**:
    - Distinguish between hard and soft skills. Prioritize technical skills relevant to the job.
    - Use concise language to present expertise (e.g., "5 years of Python experience; created software available on GitHub").

    4. **Project Section**:
   - Highlight key projects, especially those relevant to the job description.
   - Include clear objectives, tools/technologies used, methodologies, and measurable outcomes (e.g., "Developed a machine learning model with 95% accuracy to predict customer churn").
   - Focus on demonstrating problem-solving skills and technical expertise through projects.

    5. **Leadership & Activities**:
    - Include significant contributions or leadership roles in personal or extracurricular activities.
    - Link these experiences to transferable skills relevant to the job.

    6. **Formatting & Readability**:
    - Maintain professional formatting: concise bullet points, clear sections, and clean fonts.
    - Avoid excessive wordiness; focus on presenting key details succinctly.

    ### Instructions:
    Analyze the following resume text and the provided job description to suggest improvements. Rewrite sections where necessary to enhance clarity, alignment, and professionalism. Provide your suggestions in the specified table format.

    **Resume:** {resume_text}
    **Job Description:** {state['job_description']}

    ### Output Format:
    Provide suggestions in a Markdown table with the following structure:

    | Section           | Before                        | After                         | Explanation                     |
    |-------------------|------------------------------|------------------------------|---------------------------------|
    | *Section Name*   | *Original text goes here*   | *Rewritten text goes here*  | *Reason for changes*          |
    | *Section Name*   | *Original text goes here*   | *Rewritten text goes here*  | *Reason for changes*          |
    | ...             | ...                          | ...                          | ...                            |

    ### Notes:
    - Ensure the suggestions are professional, concise, and ATS-friendly.
    - Tailor each rewrite to make the resume more appealing to the specific job description.
    - Retain the original formatting style (e.g., bullet points) where appropriate.
    """

    try:
        response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
        suggestions_table = response.text.strip()
        state.setdefault("individual_scores", {})["Rewrite Suggestions"] = suggestions_table
    except Exception as e:
        state.setdefault("individual_scores", {})["Rewrite Suggestions"] = f"Error: {str(e)}"

def rank_resumes_with_llm(resumes_for_ranking, jd):
    prompt = f"""
    You are tasked with ranking a list of resumes based on their relevance to the provided job description (JD).
    Evaluate each resume’s match to the JD and rank them accordingly. Format the results in a table.

    **Job Description:** 
    {jd}

    **Resumes:**
    {resumes_for_ranking}

    **Output Format:**
    Create a table with the following columns:
    - **Rank**: The ranking position of the resume.
    - **Resume Name**: The name of the resume file.
    - **Overall Score (%)**: The overall score for the resume's match to the JD.
    - **Skills Score (%)**: Score for the skills section.
    - **Projects Score (%)**: Score for the projects section.
    - **Experience Score (%)**: Score for the experience section.

    **Example Table:**
    

    | Rank | Resume Name       | Overall Score (%) | Skills Score (%) | Projects Score (%) | Experience Score (%) |
    |------|-------------------|-------------------|------------------|--------------------|----------------------|
    | 1    | Resume1.pdf       | 92                | 90               | 88                 | 96                   | 
    | 2    | Resume2.pdf       | 85                | 83               | 80                 | 92                   |
    | 3    | Resume3.pdf       | 65                | 70               | 65                 | 60                   |
    | 4    | Resume4.pdf       | 74                | 82               | 69                 | 73                   |
    | 5    | Resume5.pdf       | 85                | 80               | 83                 | 92                   | 

    Ensure the table is well-structured and easy to understand. Use markdown or plain text formatting for the table.
    

    **Criteria for Scoring and Ranking:**
    For each resume, provide a detailed explanation of why the resume received the given score, based on the JD’s requirements.

    - Consider whether the resume fully aligns with the JD’s required skills, experiences, and projects.
    - Identify any key gaps or discrepancies in the resumes relative to the JD.
    - Consider overall relevance of the resume’s skills, projects, and experience sections to the JD.
    - Identify gaps or areas for improvement in each resume.
    - Provide clear reasoning for each ranking position.

    **Scoring Methodology:**
    - The **Overall Score** is calculated as the simple average of the following:
    - **Skills Score**
    - **Projects Score**
    - **Experience Score**

    Formula for calculating **Overall Score**:
    Overall Score = (Skills Score + Projects Score + Experience Score) / 3

    For example, if a resume has the following scores:
    - Skills Score = 80%
    - Projects Score = 60%
    - Experience Score = 70%

    The Overall Score would be calculated as:
    Overall Score = (80 + 60 + 70) / 3 = 210 / 3 = 70%


    Be sure that the explanations are clear, detailed, and separate from the table.
    """
    
    try:
        response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"
    
# Streamlit App
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
        
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f4f4f9;
        }

        .header {
            font-size: 40px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }

        .subheader {
            font-size: 18px;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 50px;
        }

        .stButton>button {
            background-color: #3498db;
            color: white;
            font-size: 16px;
            border-radius: 12px;
            padding: 10px 20px;
        }

        .stTextInput input {
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
        }

        .stFileUploader {
            font-size: 16px;
            padding: 10px;
            border-radius: 10px;
            background-color: #ecf0f1;
        }

        .stExpanderHeader {
            font-size: 22px;
            color: #34495e;
        }

        .footer {
            font-size: 14px;
            color: #95a5a6;
            text-align: center;
            margin-top: 50px;
        }

        .card {
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            margin-bottom: 20px;
            padding: 20px;
        
        .main {
           max-width: 1200px; /* Adjust the width as needed */
           margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# Header and Subheader
st.markdown("""
    <div class="header">Resume Evaluator and Ranking</div>
    <div class="subheader">Evaluate resumes against job descriptions using Generative AI</div>
""", unsafe_allow_html=True)

# Sidebar for Job Description Input and Resume Upload
with st.sidebar:
    st.title("Job Description Input")
    jd_input_type = st.radio("Job Description Source:", ("Paste Job Description", "Select Benchmark JD"))
    jd = st.text_area("Paste the Job Description") if jd_input_type == "Paste Job Description" else ""
    
    if jd_input_type == "Select Benchmark JD":
        jd_options = ["Data Analyst", "Frontend Developer", "Backend Developer", "Full-stack Developer", "AI Engineer"]
        selected_jd = st.selectbox("Select Benchmark Job Description:", jd_options)
        jd_file_path = {
            "Data Analyst": r"F:\Entrans\New folder\Resume evaluator\JD\data analyst jd.txt",
            "Frontend Developer": r"F:\Entrans\New folder\Resume evaluator\JD\frontend developer jd.txt",
            "Backend Developer": r"F:\Entrans\New folder\Resume evaluator\JD\back end developer jd.txt",
            "Full-stack Developer": r"F:\Entrans\New folder\Resume evaluator\JD\full stack developer jd.txt",
            "AI Engineer": r"F:\Entrans\New folder\Resume evaluator\JD\ai engineer jd.txt"
        }[selected_jd]
        with open(jd_file_path, "r") as f:
            jd = f.read()

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

submit = st.button("Submit")

if submit:
    if uploaded_files and jd:

        if len(uploaded_files) == 1:  # Check if only one resume is uploaded
            uploaded_file = uploaded_files[0]
            reader = pdf.PdfReader(uploaded_file)
            resume_text = "".join(page.extract_text() for page in reader.pages)

            config = {"resume_text": resume_text, "job_description": jd, "scores": [], "individual_scores": {}}
            graph_builder = StateGraph(State)

            # Add nodes
            graph_builder.add_node("skill_analysis", partial(skill_analysis, resume_text=resume_text))
            graph_builder.add_node("project_analysis", partial(project_analysis, resume_text=resume_text))
            graph_builder.add_node("experience_analysis", partial(experience_analysis, resume_text=resume_text))
            graph_builder.add_node("rewrite_suggestions", partial(rewrite_suggestions, resume_text=resume_text))

            # Define graph flow
            graph_builder.set_entry_point("skill_analysis")
            graph_builder.add_edge("skill_analysis", "project_analysis")
            graph_builder.add_edge("project_analysis", "experience_analysis")
            graph_builder.add_edge("experience_analysis", "rewrite_suggestions")

            graph = graph_builder.compile()
            final_state = graph.invoke(config)

            # Display section scores and overall score in expanders for single resume
            st.write(f"### **Resume:** {uploaded_file.name}")  # Resume filename as a heading
            
            # Individual Expanders for Sections:
            for section in ["Skills Match", "Project Analysis", "Experience Analysis"]:
                with st.expander(section):  
                    st.write(final_state["individual_scores"][section])

            # Rewrite Suggestions Table:
            st.write("### Rewrite Suggestions:")
            st.markdown(final_state["individual_scores"]["Rewrite Suggestions"], unsafe_allow_html=True)
                
        else :
            resume_scores = []
            resumes_for_ranking = "" 
            for uploaded_file in uploaded_files:
                reader = pdf.PdfReader(uploaded_file)
                resume_text = "".join(page.extract_text() for page in reader.pages)

                config = {"resume_text": resume_text, "job_description": jd, "scores": [], "individual_scores": {}}
                graph_builder = StateGraph(State)

                # Add nodes
                graph_builder.add_node("skill_analysis", partial(skill_analysis, resume_text=resume_text))
                graph_builder.add_node("project_analysis", partial(project_analysis, resume_text=resume_text))
                graph_builder.add_node("experience_analysis", partial(experience_analysis, resume_text=resume_text))
            
            
                # Define graph flow
                graph_builder.set_entry_point("skill_analysis")
                graph_builder.add_edge("skill_analysis", "project_analysis")
                graph_builder.add_edge("project_analysis", "experience_analysis")
                
            
                graph = graph_builder.compile()
                final_state = graph.invoke(config)

                    
                resumes_for_ranking += f"Resume {uploaded_file.name}:\n{final_state['individual_scores']}\n\n"
        
            # Rank resumes using LLM
            ranked_resumes = rank_resumes_with_llm(resumes_for_ranking , jd)

            # Display the ranked resumes
            st.write("## **Ranked Resumes:**")
            st.write(ranked_resumes)

        
    else:
        st.warning("Please upload resumes and provide a job description.")

# Footer
st.markdown("""
    <div class="footer">Powered by Generative AI | 2025</div>
""", unsafe_allow_html=True)
