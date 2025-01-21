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

st.set_page_config(
    page_title="AI Resume Evaluator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        Evaluate the skills based on the following criteria:
        **Skills Match Evaluation**
        1. Relevance to JD: [Score]%
            - **Matched Skills**: List all explicitly mentioned skills in the resume that match the JD, including specific technologies, frameworks, and methodologies (use commas for separation).
            - **Missing Skills**: Identify essential skills, technologies, or qualifications mentioned in the JD that are absent or underrepresented in the resume (use commas for separation).
        2. Specificity and Depth: [Score]%
            - **Description**: Evaluate how specifically and thoroughly the skills are described. Include details like experience level, proficiency, and usage in context (e.g., "Advanced Python programming with 5 years of experience").
        3. Relevance to Role and Industry: [Score]%
            - **Description**: Assess how well the listed skills align with the role and industry requirements outlined in the JD, with a focus on industry-standard tools and methodologies.
        4. Level of Expertise: [Rating]/5
            - **Description**: Rate the proficiency and expertise demonstrated in each skill listed on the resume (e.g., beginner, intermediate, advanced). Provide reasoning based on the description in the resume.
        5. Overall Skills Match Score: [Score]% 
            - **Summary**: Provide a concise overview of the resume’s skills match against the JD, noting strengths and areas for improvement.
        Resume: {resume_text}
        Job Description: {state['job_description']}
        ### Output Format:
        | Section                          | Score    | Explanation                                                                                               |
        |----------------------------------|----------|-----------------------------------------------------------------------------------------------------------|
        | **Skills Match Evaluation**      | [Score]% | Matched Skills:                                                                                           |
        |                                  |          | *List of matched skills*                                                                                 |
        |                                  |          | Missing Skills:                                                                                           |
        |                                  |          | *List of missing skills*                                                                                 |
        | **Specificity and Depth**        | [Score]% | *Detailed description of how well the skills are described (e.g., experience level, proficiency).*        |
        | **Relevance to Role and Industry** | [Score]% | *Explanation of how the skills align with the role and industry requirements.*                           |
        | **Level of Expertise**           | [Rating]/5 | *Assessment of the proficiency level of each skill (e.g., beginner, intermediate, advanced).*            |
        | **Overall Skills Match Score**   | [Score]% | *Concise summary of the skills match against the JD with strengths and areas for improvement.*           |
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
        Evaluate the match based on the following criteria:
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
        Resume: {resume_text}
        Job Description: {state['job_description']}
        ### Output Format:
        | Section                          | Score    | Explanation                                                                                               |
        |----------------------------------|----------|-----------------------------------------------------------------------------------------------------------|
        | **Project Analysis Evaluation**  | [Score]% | Matched Projects:                                                                                        |
        |                                  |          | *List of matched projects*                                                                               |
        |                                  |          | Missing Projects:                                                                                        |
        |                                  |          | *List of missing projects*                                                                               |
        | **Coverage of Essential Project Experience** | [Score]% | *Detailed assessment of how well the projects cover the JD’s critical requirements and responsibilities.* |
        | **Specificity and Detail**       | [Score]% | *Evaluation of the detail level about projects, including outcomes, results, and technologies used.*      |
        | **Level of Expertise**           | [Rating]/5 | *Rating of the expertise demonstrated in the projects, considering complexity, scale, and impact.*       |
        | **Overall Project Match Score**  | [Score]% | *Summary of the overall project match with the JD, noting strengths and areas for improvement.*          |
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
        Evaluate the match based on the following criteria:
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
        Resume: {resume_text}
        Job Description: {state['job_description']}
        
        ### Output Format:
        | Section                          | Score    | Explanation                                                                                               |
        |-----------------------------------|----------|-----------------------------------------------------------------------------------------------------------|
        | **Experience Analysis Evaluation**| [Score]% | Matched Experience:                                                                                       |
        |                                   |          | *List of matched experiences*                                                                             |
        |                                   |          | Missing Experience:                                                                                       |
        |                                   |          | *List of missing experiences*                                                                             |
        | **Coverage of Essential Experience** | [Score]% | *Evaluation of how well the experience section covers the JD’s critical aspects, like responsibilities.*  |
        | **Specificity and Impact**       | [Score]% | *Assessment of the specificity of role descriptions, including measurable achievements and contributions.* |
        | **Level of Expertise**           | [Rating]/5 | *Rating of the expertise demonstrated in the experience section, considering task complexity and responsibility.* |
        | **Overall Experience Match Score** | [Score]% | *Summary of the overall experience match with the JD, noting strengths and areas for improvement.*         |
        """
        try:
            response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
            breakdown = response.text.strip()
            state.setdefault("individual_scores", {})["Experience Analysis"] = breakdown
        except Exception as e:
            state.setdefault("individual_scores", {})["Experience Analysis"] = f"Error: {str(e)}"

def rewrite_suggestions(state, resume_text):
        prompt = f"""
        You are a highly skilled resume writer and editor. Your task is to analyze the provided resume text and suggest improvements for clarity, grammar, and relevance. Focus on rewriting poorly explained sections to make them more impactful and aligned with the job description (JD).
        **Resume Rewriting and Suggestions**
        1. **Education Section**: 
        - Identify any missing details (e.g., school name, degree, year of graduation, honors).
        - Ensure the section is in reverse chronological order.
        - Suggest improvements for readability, consistency, and relevance to the job description.
        2. **Experience Section**:
        - Highlight vague or generic phrases and rewrite them with quantifiable achievements or results (e.g., "Increased sales by 20%").
        - Use action verbs and relevant industry keywords from the job description to align with the role.
        - Emphasize skills and accomplishments that are most important for the job.
        3. **Skills Section**:
        - Suggest ways to organize technical and soft skills effectively.
        - Recommend showcasing proficiency levels or certifications where applicable.
        - Ensure the skills match the key requirements in the job description.
        4. **Project Section**:
        - Highlight key projects that are most relevant to the job description.
        - Include clear objectives, tools/technologies used, methodologies, and measurable outcomes (e.g., "Developed a machine learning model with 95% accuracy to predict customer churn").
        - Focus on demonstrating problem-solving skills and technical expertise through projects.
        - Ensure that each project description is concise, impactful, and aligned with the job role.
        5. **General Improvements**:
        - Provide grammar corrections, clarity improvements, and formatting suggestions.
        - Suggest reordering or adding new sections if relevant to the job description (e.g., Certifications, Publications, Professional Development).
        - Ensure the resume is ATS-friendly by avoiding overly complex formatting, jargon, or non-standard elements.
        **Example**:
        **Before**: "Worked on data visualization."
        **After**: "Designed interactive dashboards using Tableau, enabling actionable insights for business teams, resulting in a 15% improvement in decision-making speed."
        **Resume**: {resume_text}
        **Job Description**: {state['job_description']}
        ### Output Format:
        Provide your analysis in the following EXACT format, including all section headers and subheaders:
        ## EDUCATION SECTION
        ### Befor Text:
        [Extract and quote the exact education section text]
        ### After Text:
        [Provide the rewritten version]
        ### Reasoning:
        [Explain the improvements made]
        ## EXPERIENCE SECTION
        ### Befor Text:
        [Extract and quote the exact experience section text]
        ### After Text:
        [Provide the rewritten version]
        ### Reasoning:
        [Explain the improvements made]
        ## SKILLS SECTION
        ### Befor Text:
        [Extract and quote the exact skills section text]
        ### After Text:
        [Provide the rewritten version]
        ### Reasoning:
        [Explain the improvements made]
        ## PROJECTS SECTION
        ### Befor Text:
        [Extract and quote the exact projects section text]
        ### After Text:
        [Provide the rewritten version]
        ### Reasoning:
        [Explain the improvements made]
        ## General Improvements
        ### Grammar/Clarity Suggestions:
        -[List of specific improvements made for grammar and clarity]
        ### Formatting Suggestions:
        -[List of formatting changes or additions for better readability or ATS-friendliness]
        ### Additional Suggestions:
        -[Any other improvements, such as reordering sections or adding missing information]
        IMPORTANT GUIDELINES:
        1. Always maintain this exact format with these exact headers
        2. Always include all sections even if the original resume lacks them
        3. For missing sections, indicate "Section not present in original resume" under Current Text
        4. Use bullet points consistently as shown above
        5. Keep suggestions specific and actionable
        6. Focus on alignment with the provided job description
        7. Include quantifiable achievements where possible
        """
        try:
            response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
            suggestions = response.text.strip()
            state.setdefault("individual_scores", {})["Rewrite Suggestions"] = suggestions
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
        /* Main page styling */
        .main {
            background-color: #f8f9fa;
            padding: 2rem;
        }
        
        /* Headers */
        h1 {
            color: #1e3d59;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            padding: 1rem 0;
            text-align: center;
            background: linear-gradient(90deg, #1e3d59, #17c3b2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        h2, h3 {
            color: #1e3d59;
            font-family: 'Helvetica Neue', sans-serif;
            margin-top: 2rem;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f1f3f8;
            padding: 2rem 1rem;
        }
        
        /* File uploader */
        .stFileUploader {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(90deg, #1e3d59, #17c3b2);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 50px;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Card styling for results */
        .css-1r6slb0 {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Text area styling */
        .stTextArea textarea {
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            padding: 1rem;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background-color: #17c3b2;
        }
        
        /* Custom container for results */
        .results-container {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.9rem;
            margin-top: 4rem;
            border-top: 1px solid #eee;
        }
        
        /* Loading animation */
        .stSpinner {
            text-align: center;
            color: #17c3b2;
        }
        </style>
    """, unsafe_allow_html=True)

# Header and Subheader
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>📄 AI Resume Evaluator</h1>
        <p style='font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
            Evaluate and rank resumes against job descriptions using advanced AI
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for Job Description Input and Resume Upload
with st.sidebar:
    st.title(" 📋 Job Description")
    jd_input_type = st.radio("Job Description Source:", ("Paste Job Description", "Select Benchmark JD"))
    jd = st.text_area("Paste the Job Description") if jd_input_type == "Paste Job Description" else ""
    
    if jd_input_type == "Select Benchmark JD":
        jd_options = ["Data Analyst", "Frontend Developer", "Backend Developer", "Full-stack Developer", "AI Engineer"]
        selected_jd = st.selectbox("Select Benchmark Job Description:", jd_options)
        jd_file_path = {
            "Data Analyst": r"data analyst jd.txt",
            "Frontend Developer": r"frontend developer jd.txt",
            "Backend Developer": r"back end developer jd.txt",
            "Full-stack Developer": r"full stack developer jd.txt",
            "AI Engineer": r"ai engineer jd.txt"
        }[selected_jd]
        with open(jd_file_path, "r",encoding="utf-8", errors="ignore") as f:
            jd = f.read()

st.markdown("### 📤 Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload PDF Resumes",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more resumes in PDF format"
)
submit = st.button("🚀 Analyze Resumes", use_container_width=True)

if submit:
    if uploaded_files and jd:
        with st.spinner("🔄 Analyzing resumes... Please wait..."):

            st.markdown("### 📊 Analysis Results")

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

                with st.container():
                    st.markdown(f"#### 📄 {uploaded_files[0].name}")
                    
                    # Create tabs for different analyses
                    tabs = st.tabs(["Skills", "Projects", "Experience", "Suggestions"])
                    
                    with tabs[0]:
                        st.markdown(final_state["individual_scores"]["Skills Match"])
                    with tabs[1]:
                        st.markdown(final_state["individual_scores"]["Project Analysis"])
                    with tabs[2]:
                        st.markdown(final_state["individual_scores"]["Experience Analysis"])
                    with tabs[3]:
                        st.markdown(final_state["individual_scores"]["Rewrite Suggestions"])
                    
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
                st.write("### 🏆 Ranked Results")
                st.write(ranked_resumes)

        
    else:
        st.warning("⚠️ Please upload resumes and provide a job description.")

# Footer
st.markdown("""
    <div class='footer'>
        <p>Powered by Generative AI | © 2025 Resume Evaluator</p>
    </div>
""", unsafe_allow_html=True)