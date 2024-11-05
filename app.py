# app.py
import streamlit as st
from model1 import process_pdf, embed_and_store_text, similarity_search, extract_answer
from langchain.text_splitter import CharacterTextSplitter

# Apply custom CSS for a prettier UI
def apply_custom_css():
    st.markdown("""<style>
        body { background-color: #f9f9f9; font-family: 'Arial', sans-serif; }
        .stButton button { background-color: #4CAF50; color: white; border-radius: 5px; padding: 0.5em 1em; }
        .stTextInput input { border: 1px solid #ddd; border-radius: 5px; padding: 0.5em; }
        .stFileUploader label { color: #4CAF50; font-weight: bold; }
        .reportview-container { padding: 2em; }
        h1, h2 { color: #4CAF50; }
        .stMarkdown { font-size: 1.1em; line-height: 1.6; }
        .uploaded_file_info { font-size: 0.9em; color: #888; }
    </style>""", unsafe_allow_html=True)

# Streamlit UI starts here
st.set_page_config(page_title="Interactive QA Bot", layout="centered", initial_sidebar_state="auto")

# Apply custom CSS for better aesthetics
apply_custom_css()

# Title and Description
st.title('📄 Interactive QA Bot with Document Upload')
st.markdown("This app allows you to upload a PDF document and ask questions based on its content.")

# File uploader for PDFs
st.markdown("### Upload your PDF document here:")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Check if a file is uploaded
if uploaded_file is not None:
    st.markdown("<p class='uploaded_file_info'>File uploaded successfully. Processing...</p>", unsafe_allow_html=True)
    
    # Process the uploaded PDF
    raw_text = process_pdf(uploaded_file)

    # Adjust chunk size and overlap for better context coverage
    textsplitter = CharacterTextSplitter(separator="\n", chunk_size=5000, chunk_overlap=500)
    texts = textsplitter.split_text(raw_text)
    
    # Embed the text chunks and store them in Pinecone
    progress = st.progress(0)
    embed_and_store_text(texts)
    st.success("PDF processed and stored in Pinecone.")
else:
    st.warning("Please upload a PDF document to begin.")

# User input for query
st.markdown("### Ask a question:")
query = st.text_input("Enter your question here:")

# Display the answer if query is provided
if query:
    with st.spinner("Searching for the answer..."):
        docs = similarity_search(query)
        
        if docs:
            answer = extract_answer(query, docs)
            if answer:
                st.success(f"Answer: {answer}")
            else:
                st.error("Sorry, I can't find the answer to your question.")
        else:
            st.warning("No matching documents found.")
