import streamlit as st
from model import process_pdf, embed_and_store_text_parallel, similarity_search, extract_answer
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

# File uploader for PDFs
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Process PDF and display progress
if uploaded_file is not None:
    raw_text = process_pdf(uploaded_file)
    if raw_text:
        st.progress(30)
        textsplitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        texts = textsplitter.split_text(raw_text)
        embed_and_store_text_parallel(texts)
        st.success("PDF processed and stored in Pinecone.")
        st.progress(100)
    else:
        st.error("Failed to process PDF.")

# User input for query
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




