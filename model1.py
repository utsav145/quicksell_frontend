import os
import re
import cohere
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
import logging
from concurrent.futures import ThreadPoolExecutor

# Initialize Cohere client securely
cohere_api_key = os.getenv("COHERE_API_KEY")  # Get from environment
co = cohere.Client(cohere_api_key)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = 'cohere-small-embeddings'
embedding_dimension = 4096  # Update this to match the dimension from Cohere

# Initialize index or create if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)

# Load QA model with exception handling
try:
    qa_model = pipeline("question-answering", model="deepset/roberta-large-squad2")
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a space
    text = re.sub(r'\s{2,}', ' ', text)  # Remove extra spaces
    text = ''.join(filter(lambda x: x.isprintable(), text))  # Remove non-printable characters
    return text.strip()

def embed_and_store_text_parallel(texts):
    with ThreadPoolExecutor() as executor:
        executor.map(store_embedding_in_pinecone, texts)

def store_embedding_in_pinecone(chunk, i):
    embedding = co.embed(texts=[chunk]).embeddings[0]
    metadata = {'text': chunk}
    index.upsert(vectors=[(f"doc_{i}", embedding, metadata)])

# Function to process PDF and apply preprocessing
def process_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        raw_text = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
        if not raw_text.strip():
            raise ValueError("No readable text found in the uploaded PDF.")
        
        # Preprocess the extracted text
        return preprocess_text(raw_text)  # Apply preprocess_text here
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        return None
