# AI Resume Evaluator : https://huggingface.co/spaces/ainerd11/ai_resume_evaluator


# AI Resume Evaluator

## 📄 Overview
AI Resume Evaluator is a Streamlit-based web application that uses Google's Generative AI (Gemini) to analyze and evaluate resumes against job descriptions. The application helps job seekers optimize their resumes and assists recruiters in ranking multiple candidates efficiently.


## ✨ Features
- **Single Resume Analysis**: Detailed evaluation of a resume against a job description
  - Skills match analysis
  - Project relevance analysis
  - Experience evaluation
  - Rewrite suggestions for improvement
- **Multiple Resume Ranking**: Compare and rank multiple resumes against a job description
- **Benchmark Job Descriptions**: Choose from pre-defined job descriptions for common roles
- **Interactive UI**: User-friendly interface with tabs for different analysis components
- **PDF Support**: Directly upload and analyze PDF resumes

## 🛠️ Technology Stack
- **Frontend**: Streamlit
- **AI Model**: Google Generative AI (Gemini 2.0 Flash)
- **PDF Processing**: PyPDF2
- **Workflow Orchestration**: LangGraph
- **Environment Management**: dotenv

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Google Generative AI API key

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-resume-evaluator.git
   cd ai-resume-evaluator
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 📋 Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Access the application in your web browser (typically at http://localhost:8501).

3. In the sidebar:
   - Input a job description by pasting it directly or selecting a benchmark JD
   
4. In the main panel:
   - Upload one or more resume PDFs
   - Click "Analyze Resumes" to start the evaluation process

5. View the results:
   - For a single resume: Navigate through tabs to see skills analysis, project analysis, experience analysis, and rewrite suggestions
   - For multiple resumes: View the ranked results table with scores and explanations

## 🔄 Workflow

The application uses a LangGraph-powered workflow with the following stages:
1. **Skills Analysis**: Evaluates how well the candidate's skills match the job requirements
2. **Project Analysis**: Assesses the relevance and impact of the candidate's projects
3. **Experience Analysis**: Analyzes work experience against job requirements
4. **Rewrite Suggestions** (for single resume): Provides detailed recommendations for improving the resume

## 📁 Project Structure

```
ai-resume-evaluator/
├── app.py                 # Main application file
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (not in repository)
├── benchmark_jds/         # Folder containing benchmark job descriptions
│   ├── data_analyst.txt
│   ├── frontend_developer.txt
│   ├── backend_developer.txt
│   ├── fullstack_developer.txt
│   └── ai_engineer.txt
└── README.md              # Project documentation
```

## ⚙️ Configuration

You can customize the application by:
- Adding more benchmark job descriptions in the benchmark_jds folder
- Modifying the prompts in the analysis functions to adjust evaluation criteria
- Updating the UI styling in the Streamlit CSS section

## 🔒 Security Considerations

- The application processes resume data locally and does not store any user data
- Job descriptions and resumes are sent to Google's Generative AI for analysis
- Ensure you follow Google's API usage policies and data protection regulations


## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Google Generative AI](https://ai.google.dev/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [PyPDF2](https://pypdf2.readthedocs.io/)
