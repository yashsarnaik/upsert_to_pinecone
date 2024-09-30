import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
import time
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import json
from PyPDF2 import PdfReader
from langchain.schema import Document

st.set_page_config(
    page_title="Upsert to Pinecone",
    page_icon="📤")

def load_css(file_path):
    with open(file_path, "r") as f:
        return f"<style>{f.read()}</style>"

# Load and inject CSS
css = load_css("style.css")
st.markdown(css, unsafe_allow_html=True)

# Load environment variables
load_dotenv()

st.title('Upsert to Pinecone📤')

# File uploader for Google Credentials
credential_path = st.file_uploader("Choose a Google Credentials JSON file🗄️", type="json")

if credential_path is not None:
    credentials = json.load(credential_path)
    temp_credentials_path = "temp_credentials.json"
    
    with open(temp_credentials_path, "w") as f:
        json.dump(credentials, f)
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_credentials_path
    st.success("Google Credentials uploaded and environment variable set successfully!")

# PDF file uploader
uploaded_file = st.file_uploader("Choose a PDF file📁", type="pdf")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings.embed_documents(text_chunks)


def get_vectorstore(text_chunks, index_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create Document objects
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    
    # Create and return the vector store
    vectorstore = LangchainPinecone.from_documents(
        documents,
        embeddings,
        index_name=index_name
    )
    
    return vectorstore

# Pinecone setup
key = st.text_input("Enter your Pinecone API key🔑", type="password")
index_name = st.text_input("Enter your Pinecone Index Name📛")

if key and index_name:
    # Set the Pinecone API key as an environment variable
    os.environ['PINECONE_API_KEY'] = key

    # Initialize Pinecone
    pc = Pinecone()
    spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
    )

    # Check if the index exists, if not create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,  # Dimension for Google's embedding-001 model
            metric='cosine',
            spec=spec
        )
        st.info(f"Created new Pinecone index: {index_name}")

    # Get the index
    index = pc.Index(index_name)

    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        text_chunks = get_text_chunks(text)
        
        if st.button("Generate Embeddings and Create Vectorstore"):
            with st.spinner("Processing..."):
                embeddings = get_embeddings(text_chunks)
                vectorstore = get_vectorstore(text_chunks, index_name)
            
            st.success("Embeddings generated and vectorstore created successfully!")
            st.write(f"Number of chunks: {len(text_chunks)}")
            st.write(f"Embedding dimension: {len(embeddings[0])}")
            
    

            # You can add more functionality here, such as querying the vectorstore
else:
    st.warning("Please enter your Pinecone API key and Index Name to proceed.")

# Clean up the temporary credentials file
if os.path.exists("temp_credentials.json"):
    os.remove("temp_credentials.json")

footer=("""
1. Download the Default credentials JSON file from Google Cloud Console.
2. Upload the PDF file you want to vectorize and upload to the Pinecone Database.
3. Enter your Pinecone API key.
4. Enter your Pinecone Index name.
5. Selected environment by default is <h3> us-east-1 </h3> if you want a different one make changes in app.py.             
""")    

st.markdown(footer,unsafe_allow_html=True)
