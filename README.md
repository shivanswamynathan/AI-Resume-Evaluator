# AI Resume Evaluator : https://huggingface.co/spaces/ainerd11/ai_resume_evaluator


## Project Overview
This project is an AI-powered resume evaluator built using Python and Streamlit. It leverages Google's Gemini large language model (LLM) to analyze resumes against job descriptions, providing detailed feedback and rankings.

## Key Features
- **Resume Analysis**: Provides an in-depth evaluation of resumes, focusing on skills, projects, and experience. It identifies areas of strength and weakness in comparison to the job description.
- **Resume Ranking**: Ranks multiple resumes based on their relevance to the job description, streamlining the candidate selection process.
- **Rewrite Suggestions**: Offers actionable suggestions to improve resumes, focusing on clarity, grammar, and alignment with the job description.
- **User-Friendly Interface**: Designed with Streamlit for an intuitive web application accessible to both recruiters and job seekers.

## Technical Details

### 1. Libraries and Dependencies
- **PyPDF2**: Extracts text from PDF resumes for analysis.
- **Streamlit**: Powers the interactive web interface for user input and results display.
- **google-generativeai**: Connects to Google's Gemini LLM for advanced natural language processing and analysis.
- **langgraph**: Manages the workflow of resume evaluation using a directed acyclic graph (DAG).
- **dotenv**: Loads environment variables securely, including the Gemini API key.
- **functools**: Simplifies function management with `partial` for pre-filling arguments.
- **typing**: Enhances code clarity with type annotations like `TypedDict` and `Annotated`.

### 2. Core Components
#### Analysis Functions
- **`skill_analysis`**: Evaluates skills for relevance, depth, and proficiency based on the job description.
- **`project_analysis`**: Assesses the alignment and quality of projects with the job requirements.
- **`experience_analysis`**: Reviews work experience, focusing on relevance, impact, and expertise.
- **`rewrite_suggestions`**: Provides detailed rewriting suggestions to improve the resume's content and clarity.
- **`rank_resumes_with_llm`**: Ranks resumes based on overall relevance to the job description.

#### State Management
- The `State` class (`TypedDict`) organizes and stores data, ensuring smooth communication between analysis functions.

#### Prompt Engineering
- Carefully crafted prompts guide Gemini to generate precise, actionable feedback for resumes.

### 3. Workflow and Execution
1. **Initialization**: Streamlit sets up the user interface.
2. **Job Description Input**: Users can paste or select a job description.
3. **Resume Upload**: Users upload PDF resumes.
4. **Analysis Trigger**: Clicking "🚀 Analyze Resumes" starts the analysis process.
5. **Text Extraction**: PyPDF2 extracts text from resumes for analysis.
6. **Analysis Execution**: 
   - Single Resume: Executes analysis functions sequentially using `langgraph`.
   - Multiple Resumes: Processes each resume, then ranks them using Gemini.
7. **Results Display**: Presents results in tabs for individual resumes or as a ranked list for multiple resumes.

### 4. Configuration and Environment
- **API Key**: Stored securely in a `.env` file for interacting with the Gemini LLM.

## Usage Instructions

### 1.  Launching the Application:
-Follow the setup instructions to install libraries, obtain an API key, and create the .env file.
-Run the application using the command: streamlit run your_script_name.py.
-This will open the AI Resume Evaluator application in your web browser.

## 2. Providing a Job Description:

In the sidebar, you'll find a section labeled **"Job Description."**

You have two options:

- **Paste Job Description:** Directly paste the text of the job description into the text area provided.
- **Select Benchmark JD:** Choose a pre-defined job description from the dropdown menu (e.g., Data Analyst, Frontend Developer). This option is useful for quick testing or when you want to use a standardized job description.

## 3. Uploading Resumes:

In the main section of the application, you'll see the **"Upload Resumes"** area.

- Click the **"Browse files"** button to select the PDF resume(s) you want to analyze from your computer.
- You can upload either a single resume or multiple resumes at once.

## 4. Initiating the Analysis:

Once you've provided a job description and uploaded resumes, click the **"🚀 Analyze Resumes"** button.

- The application will begin processing the resumes and performing the analysis. This may take a few moments, depending on the complexity of the resumes and the load on the Google Gemini API.

## 5. Reviewing the Results:

### Scenario 1: Single Resume Analysis

If you uploaded one resume, the results will be presented in a structured format with separate tabs:

- **Skills:** This tab displays the analysis of the skills mentioned in the resume, including matched skills, missing skills, proficiency levels, and an overall score for the skills section.
- **Projects:** This tab presents the evaluation of projects listed in the resume, highlighting relevant projects, missing project types, and an overall score for the project section.
- **Experience:** This tab provides insights into the candidate's work experience, emphasizing alignment with the job description and areas for improvement.
- **Suggestions:** This tab offers concrete recommendations for improving the resume, such as rewriting vague phrases, incorporating action verbs, and highlighting key achievements.

### Scenario 2: Multiple Resume Ranking

If you uploaded more than one resume, the application will perform a ranking analysis.

The results will be displayed in a table that includes:

- **Rank:** The overall ranking of each resume based on its relevance to the job description.
- **Resume Name:** The file name of each uploaded resume.
- **Overall Score (%):** A composite score representing the overall match of the resume to the job description.
- **Skills Score (%):** The score for the skills section.
- **Projects Score (%):** The score for the projects section.
- **Experience Score (%):** The score for the experience section.

In addition to the table, the application will provide explanations for the ranking, highlighting the key factors that contributed to the placement of each resume.

## Important Considerations:

- **Interpreting Results:** The AI-powered analysis provides valuable insights, but it's essential to remember that it is a tool to support human judgment. Carefully review the results and consider them in conjunction with your own assessment of the candidates.
- **Privacy and Security:** Ensure that you handle resumes and job descriptions responsibly and in accordance with relevant data privacy regulations.

By following these detailed usage instructions, you can effectively utilize the AI Resume Evaluator to analyze and rank resumes, making your candidate selection process more efficient and informed.

