# Initialize the retriever and Groq client
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = Groq(api_key="Your API Key here")

# Knowledge base (documents) and embeddings
documents = [
    "Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths of retrieval-based and generative models.",
    "The main components of a RAG system are the retriever and the generator.",
    "A key benefit of Retrieval-Augmented Generation is that it can produce more accurate responses compared to standalone generative models.",
    "The retrieval process in a RAG system often relies on embedding-based models, like Sentence-BERT or DPR.",
    "Common use cases of RAG include chatbots, customer support systems, and knowledge retrieval for business intelligence."
]
document_embeddings = retriever.encode(documents, convert_to_tensor=True)

# Function to retrieve top relevant document
def retrieve(query, top_k=1):
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, document_embeddings, top_k=top_k)
    top_docs = [documents[hit['corpus_id']] for hit in hits[0]]
    return top_docs[0] if hits[0] else None

# Function to generate response using Groq
def generate_response(query, context):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Context: {context} Question: {query} Answer:"
            }
        ],
        model="gemma2-9b-it"
    )
    return response.choices[0].message.content

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
    document_embeddings = retriever.encode(documents, convert_to_tensor=True)

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
        answer = generate_response(question, retrieved_context)
    else:
        answer = "I have no knowledge about this topic."
    st.write("Answer:", answer)
