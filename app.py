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
st.markdown("This app allows you to upload a PDF document and ask questions based on its content. It maintains context to enable a continuous conversation.")

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

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
    embed_and_store_text(texts)
    st.success("PDF processed and stored in Pinecone.")
else:
    st.warning("Please upload a PDF document to begin.")

# Sidebar for conversation history
with st.sidebar:
    st.markdown("### Conversation History")
    if st.session_state.conversation_history:
        for entry in st.session_state.conversation_history:
            st.markdown(f"**You:** {entry['question']}")
            st.markdown(f"**Answer:** {entry['answer']}")
    else:
        st.write("No conversation history yet.")

# User input for query
st.markdown("### Ask a question:")
query = st.text_input("Enter your question here:", key="question_input")

# Handle user query
if query:
    with st.spinner("Searching for the answer..."):
        docs = similarity_search(query)

        if docs:
            answer = extract_answer(query, docs)
            if answer:
                # Append to conversation history
                st.session_state.conversation_history.append({"question": query, "answer": answer})
                st.success(f"Answer: {answer}")
            else:
                st.session_state.conversation_history.append({"question": query, "answer": "Sorry, I can't find the answer to your question."})
                st.error("Sorry, I can't find the answer to your question.")
        else:
            st.session_state.conversation_history.append({"question": query, "answer": "No matching documents found."})
            st.warning("No matching documents found.")
