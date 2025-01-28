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

def profile_summary_analysis(state, resume_text):
    prompt = f"""
    Analyze the profile summary of the resume and provide brief, focused feedback (maximum 500 words):

    ### Current Summary Analysis
    [Brief description of current summary - 2-3 sentences]

    ### Key Strengths 
    - [Focus on most impactful elements (2-3 points)]

    ### Critical Improvements Needed 
    - [List only the most important changes needed (2-3 points)]

    ### Quick Recommendations
    [2-3 specific, actionable suggestions]

    Resume: {resume_text}
    Job Description: {state['job_description']}

    Keep the entire response under 500 words.
    """
    try:
        response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
        state.setdefault("individual_scores", {})["Profile Summary"] = response.text.strip()
    except Exception as e:
        state.setdefault("individual_scores", {})["Profile Summary"] = f"Error: {str(e)}"

def experience_analysis(state, resume_text):
    prompt = f"""
    Analyze the experience section of the resume against the job description and provide feedback in the following format (maximum 500 words):

    ### Experience Analysis Evaluation Table
    | Section | Score | Explanation |
    |---------|--------|-------------|
    | Relevance to JD | [Score]% | Match with job requirements |
    | Coverage of Essential Experience | [Score]% | Assessment of required experience |
    | Specificity and Impact | [Score]% | Detail and measurable achievements |
    | Level of Expertise | [Rating]/5 | Demonstrated expertise level |
    | Overall Experience Match Score | [Score]% | Final experience evaluation |

    ### Critical Gaps 
    - [List only the most important missing experience (2-3 points)]

    ### Key Pattern Suggestions 
    Consider these patterns (do not rewrite) (2-3 examples):
    - "Action + Task + Tools + Outcome" example
    - Second pattern example if needed

    ### Priority Improvements 
    - [Focus on highest-impact changes (2-3 points)]

    Resume: {resume_text}
    Job Description: {state['job_description']}

    Keep the entire response under 500 words.
    """
    try:
        response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
        breakdown = response.text.strip()
        state.setdefault("individual_scores", {})["Experience Analysis"] = breakdown
    except Exception as e:
        state.setdefault("individual_scores", {})["Experience Analysis"] = f"Error: {str(e)}"


def education_analysis(state, resume_text):
    prompt = f"""
    Provide concise feedback on the education section (maximum 500 words):

    ### Quick Assessment
    [2-3 sentences on overall education section]

    ### Key Strengths 
    - [List most relevant educational qualifications (2-3 points)]

    ### Essential Improvements 
    - [Focus on critical missing elements (2-3 points)]

    ### Priority Recommendations
    [2-3 specific suggestions for improvement]

    Resume: {resume_text}
    Job Description: {state['job_description']}

    Keep the entire response under 500 words.
    """
    try:
        response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
        state.setdefault("individual_scores", {})["Education Analysis"] = response.text.strip()
    except Exception as e:
        state.setdefault("individual_scores", {})["Education Analysis"] = f"Error: {str(e)}"



def skill_analysis(state, resume_text):
    prompt = f"""
    Analyze the skills section of the resume against the job description and provide feedback in the following format (maximum 500 words):

    ### Skills Match Evaluation Table
    | Section | Score | Explanation |
    |---------|--------|-------------|
    | Relevance to JD | [Score]% | Analysis of direct skills match |
    | Specificity and Depth | [Score]% | Assessment of skills description detail |
    | Relevance to Role and Industry | [Score]% | Alignment with industry requirements |
    | Level of Expertise | [Rating]/5 | Overall expertise demonstrated |
    | Overall Skills Match Score | [Score]% | Composite skills assessment |

    ### Critical Gaps 
    - [List only the most important missing skills (3-4 points)]

    ### Key Recommendations 
    - [Focus on highest-impact improvements (2-3 points)]

    Resume: {resume_text}
    Job Description: {state['job_description']}

    Keep the entire response under 500 words.
    """
    try:
        response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
        breakdown = response.text.strip()
        state.setdefault("individual_scores", {})["Skills Match"] = breakdown
    except Exception as e:
        state.setdefault("individual_scores", {})["Skills Match"] = f"Error: {str(e)}"

def project_analysis(state, resume_text):
    prompt = f"""
    Analyze the projects section of the resume against the job description and provide feedback in the following format (maximum 500 words):

    ### Project Analysis Evaluation Table
    | Section | Score | Explanation |
    |---------|--------|-------------|
    | Relevance to JD | [Score]% | Projects' alignment with requirements |
    | Coverage of Essential Project Experience | [Score]% | Assessment of project scope coverage |
    | Specificity and Detail | [Score]% | Level of detail in project descriptions |
    | Level of Expertise | [Rating]/5 | Technical complexity demonstrated |
    | Overall Project Match Score | [Score]% | Final project evaluation |

    ### Missing Project Types 
    - [List only essential missing projects (2-3 most critical)]

    ### Key Pattern Suggestions 
    Consider these patterns (do not rewrite) (2-3 examples):
    - "Action + Task + Tools + Outcome" example
    - Second pattern example if needed

    ### Priority Improvements 
    - [List highest-impact changes needed (2-3 points)]

    Resume: {resume_text}
    Job Description: {state['job_description']}

    Keep the entire response under 500 words.
    """
    try:
        response = genai.GenerativeModel('gemini-2.0-flash-exp').generate_content(prompt)
        breakdown = response.text.strip()
        state.setdefault("individual_scores", {})["Project Analysis"] = breakdown
    except Exception as e:
        state.setdefault("individual_scores", {})["Project Analysis"] = f"Error: {str(e)}"


def rewrite_suggestions(state, resume_text):
        prompt = f"""
        You are a highly skilled resume writer and editor. Your task is to analyze the provided resume text and suggest improvements for clarity, grammar, and relevance.
        Rewrite poorly explained sections to make them more impactful and aligned with the job description (JD). Use the "Expected Sentence Structure" for consistency across all sections.

        **Rewrite Guidelines by Section**:

        1. **Profile Summary**:
        - Ensure it is concise, uses strong action verbs, and highlights key skills, experience, and achievements.
        - Expected Sentence Structure:
            "Accomplished [Job Title/Professional], skilled in [Core Skills/Tools], with [Years of Experience] in [Industry/Domain]. Proven ability to [Specific Achievement/Impact] using [Tools/Technologies]. Seeking to [Career Goal/Objective]."

        2. **Skills Section**:
        - Recommend organizing skills into categories (e.g., Programming Languages, Tools, Frameworks).
        - Use the sentence structure:
            "Proficient in [Skill/Technology], experienced with [Tool/Framework], and familiar with [Additional Skills]."

        3. **Experience Section**:
        - Rewrite vague descriptions using measurable outcomes and specific tools. Use the structure:
            "Action Verb + Task/Feature + Tools/Technology + Measurable Outcome."
        - Example: "Implemented a machine learning pipeline using Python and TensorFlow, reducing data processing time by 30% and improving model accuracy by 15%."

        4. **Projects Section**:
        - Focus on outcomes, technologies, and measurable impacts. Use the structure:
            "Action Verb + Feature/Task + Tools/Technology + Measurable Outcome."
        - Example: "Developed a recommendation engine using Python and Scikit-learn, increasing user engagement by 25%."

        5. **Education Section**:
        - Verify completeness (degree, institution, year, relevant coursework).
        - Example: "Master’s in Data Science, XYZ University, 2022. Relevant Coursework: Machine Learning, Deep Learning, Big Data Analytics."

        6. **Achievements/Certifications**:
        - Ensure certifications and awards are relevant, recent, and properly formatted.
        - Example: "Certified Machine Learning Specialist, Coursera, 2023."

        **Example Rewriting:**

        **Before:** "Worked on data visualization."
        **After:** "Designed interactive dashboards using Tableau, enabling actionable insights for business teams."
        **Reason for Change:** The rewrite adds specific tools, quantifiable outcomes, and aligns the task with job-related keywords.

        **Resume:** {resume_text}
        **Job Description:** {state['job_description']}

        **Output Format:**
        ## Section Name: [e.g., Education, Experience, Skills]
        ### Original Text: [Provide the original resume text, if present]
        ### Suggested Rewrite: [Provide the improved text]
        ### Explanation: [Explain why the change was made and how it improves the resume]

        Ensure all suggestions are professional, concise, and tailored to the JD. Skip sections not present in the resume.
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
            "Data Analyst": r"F:\Entrans\New folder\Resume evaluator\JD\data analyst jd.txt",
            "Frontend Developer": r"F:\Entrans\New folder\Resume evaluator\JD\frontend developer jd.txt",
            "Backend Developer": r"F:\Entrans\New folder\Resume evaluator\JD\back end developer jd.txt",
            "Full-stack Developer": r"F:\Entrans\New folder\Resume evaluator\JD\full stack developer jd.txt",
            "AI Engineer": r"F:\Entrans\New folder\Resume evaluator\JD\ai engineer jd.txt"
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
                graph_builder.add_node("profile_summary_analysis", partial(profile_summary_analysis, resume_text=resume_text))
                graph_builder.add_node("experience_analysis", partial(experience_analysis, resume_text=resume_text))
                graph_builder.add_node("education_analysis", partial(education_analysis, resume_text=resume_text))
                graph_builder.add_node("skill_analysis", partial(skill_analysis, resume_text=resume_text))
                graph_builder.add_node("project_analysis", partial(project_analysis, resume_text=resume_text))
                graph_builder.add_node("rewrite_suggestions", partial(rewrite_suggestions, resume_text=resume_text))

                # Define graph flow
                graph_builder.set_entry_point("profile_summary_analysis")
                graph_builder.add_edge("profile_summary_analysis", "experience_analysis")
                graph_builder.add_edge("experience_analysis", "education_analysis")
                graph_builder.add_edge("education_analysis", "skill_analysis")
                graph_builder.add_edge("skill_analysis", "project_analysis")
                graph_builder.add_edge("project_analysis", "rewrite_suggestions")

                graph = graph_builder.compile()
                final_state = graph.invoke(config)

                with st.container():
                    st.markdown(f"#### 📄 {uploaded_files[0].name}")
                    
                    # Create tabs for different analyses
                    tabs = st.tabs(["Profile Summary", "Experience", "Education", "Skills", "Projects", "Suggestions"])
                    
                    with tabs[0]:
                        st.markdown(final_state["individual_scores"]["Profile Summary"])
                    with tabs[1]:
                        st.markdown(final_state["individual_scores"]["Experience Analysis"])
                    with tabs[2]:
                        st.markdown(final_state["individual_scores"]["Education Analysis"])
                    with tabs[3]:
                        st.markdown(final_state["individual_scores"]["Skills Match"])
                    with tabs[4]:
                        st.markdown(final_state["individual_scores"]["Project Analysis"])
                    with tabs[5]:
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
                    graph_builder.add_node("profile_summary_analysis", partial(profile_summary_analysis, resume_text=resume_text))
                    graph_builder.add_node("experience_analysis", partial(experience_analysis, resume_text=resume_text))
                    graph_builder.add_node("education_analysis", partial(education_analysis, resume_text=resume_text))
                    graph_builder.add_node("skill_analysis", partial(skill_analysis, resume_text=resume_text))
                    graph_builder.add_node("project_analysis", partial(project_analysis, resume_text=resume_text))
                
                
                    # Define graph flow
                    graph_builder.set_entry_point("profile_summary_analysis")
                    graph_builder.add_edge("profile_summary_analysis", "experience_analysis")
                    graph_builder.add_edge("experience_analysis", "education_analysis")
                    graph_builder.add_edge("education_analysis", "skill_analysis")
                    graph_builder.add_edge("skill_analysis", "project_analysis")
                    
                
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