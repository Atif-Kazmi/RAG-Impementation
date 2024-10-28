# Required imports
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PyPDF2 import PdfReader

# Initialize model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Knowledge base (documents) and embeddings
documents = [
    "Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths of retrieval-based and generative models.",
    "The main components of a RAG system are the retriever and the generator.",
    "A key benefit of Retrieval-Augmented Generation is that it can produce more accurate responses compared to standalone generative models.",
    "The retrieval process in a RAG system often relies on embedding-based models, like Sentence-BERT or DPR.",
    "Common use cases of RAG include chatbots, customer support systems, and knowledge retrieval for business intelligence."
]

# Function to encode documents
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

document_embeddings = [encode_text(doc) for doc in documents]

# Function to retrieve top relevant document
def retrieve(query, top_k=1):
    query_embedding = encode_text(query)
    similarities = cosine_similarity(query_embedding, np.vstack(document_embeddings))
    top_idx = similarities.argsort()[0][-top_k:][::-1]  # Get top_k indices
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
