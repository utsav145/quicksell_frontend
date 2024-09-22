# model.py
import os
import re
import cohere
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline

# Initialize Cohere client
cohere_api_key = "Vfrg0t3eibW8K8QCYyBcKzTho7dZIcYxGFHCrcm7"  # Replace with your actual API key
co = cohere.Client(cohere_api_key)

# Initialize Pinecone with API key and environment
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", "b4026b13-3fd6-4aea-bf3d-7e33f330a885"))
index_name = 'cohere-small-embeddings'
embedding_dimension = 4096  # Update this to match the dimension from Cohere

# Initialize index or create if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Load the larger QA model for better answers
qa_model = pipeline("question-answering", model="deepset/roberta-large-squad2")

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = ''.join(filter(lambda x: x.isprintable(), text))
    return text.strip()

# Function to perform similarity search using Pinecone
def similarity_search(query, top_k=10):
    query_embedding = co.embed(texts=[query]).embeddings[0]
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results['matches']

# Function to extract relevant answer from text
def extract_answer(query, texts):
    context = " ".join(text['metadata']['text'] for text in texts)
    result = qa_model(question=query, context=context, max_length=1000, min_length=100)
    return result['answer']

# Function to process the uploaded PDF
def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    raw_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return preprocess_text(raw_text)

# Function to embed text chunks and store them in Pinecone
def embed_and_store_text(texts):
    for i, chunk in enumerate(texts):
        embedding = co.embed(texts=[chunk]).embeddings[0]
        metadata = {'text': chunk}
        index.upsert(vectors=[(f"doc_{i}", embedding, metadata)])

