## Interactive QA Bot

## Overview
This Streamlit application allows users to upload PDF documents and ask questions based on their content. The app processes the document, extracts relevant information, and generates answers using a question-answering model.

## Features
- Upload PDF documents.
- Ask questions based on the content of the uploaded document.
- Generate answers using advanced NLP models.

## Technologies Used
- Streamlit: For building the web application interface.
- Pinecone: For vector database storage and similarity search.
- Cohere: For generating text embeddings.
- PyPDF2: For extracting text from PDF files.
- Langchain: For text splitting.
- Transformers: For question-answering model.

## Installation

### Prerequisites
- Python 3.7 or higher
- An active Cohere and Pinecone account for API keys

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
2. Install the required libraries:
  pip install -r requirements.txt

5. Set up environment variables. Create a .env file in the root directory and add your API keys:
  PINECONE_API_KEY=your_pinecone_api_key
  COHERE_API_KEY=your_cohere_api_key

# Usage:
   Run the app:
    streamlit run app.py

  Open your web browser and navigate to http://localhost:8501.

  Upload a PDF document and enter your question in the provided input field.


# Deployment
To deploy the app for public access, you can use platforms such as Streamlit Cloud or Heroku. Follow the respective documentation for deployment instructions.

# Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for details.


# Acknowledgments
Streamlit - The framework used for building the app.
Pinecone - The vector database service.
Cohere - The NLP service used for embeddings.
Transformers - The library used for question-answering models.

   
