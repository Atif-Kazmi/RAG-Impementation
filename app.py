# Required imports
import streamlit as st
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-MiniLM-L6-v2"
headers = {"Authorization": f"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"}  # Replace with your Hugging Face API token

# Knowledge base (documents) and embeddings
documents = [
    "Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths of retrieval-based and generative models.",
    "The main components of a RAG system are the retriever and the generator.",
    "A key benefit of Retrieval-Augmented Generation is that it can produce more accurate responses compared to standalone generative models.",
    "The retrieval process in a RAG system often relies on embedding-based models, like Sentence-BERT or DPR.",
    "Common use cases of RAG include chatbots, customer support systems, and knowledge retrieval for business intelligence."
]

# Function to encode text using Hugging Face Inference API
def encode_text(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}})
    embeddings = response.json()[0] if response.status_code == 200 else None
    return np.mean(embeddings, axis=0) if embeddings else None

# Precompute embeddings for documents
document_embeddings = [encode_text(doc) for doc in documents]

# Function to retrieve top relevant document
def retrieve(query, top_k=1):
    query_embedding = encode_text(query)
    if query_embedding is None:
        return "Error: Unable to retrieve embedding for query."

    similarities = cosine_similarity([query_embedding], document_embeddings)
    top_idx = similarities.argsort()[0][-top_k:][::-1]
    top_docs = [documents[idx] for idx in top_idx]
    return top_docs[0] if top_docs else None

# Function to handle PDF upload and text extraction
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to update knowledge base with new content from PDF
def update_knowledge_base(pdf_text):
    global documents, document_embeddings
    documents.append(pdf_text)
    document_embeddings.append(encode_text(pdf_text))

# Streamlit app layout
st.title("RAG-based Question Answering App")
st.write("Upload a PDF, ask questions based on its content, and get answers!")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    update_knowledge_base(pdf_text)
    st.write("PDF content successfully added to the knowledge base.")

# Question input
question = st.text_input("Enter your question:")
if question:
    retrieved_context = retrieve(question)
    if retrieved_context:
        answer = f"Context: {retrieved_context} Answer: Based on the information found."
    else:
        answer = "I have no knowledge about this topic."
    st.write("Answer:", answer)
