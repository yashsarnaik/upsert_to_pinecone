
# Upsert to Pinecone

This Streamlit app allows you to easily upload PDFs, vectorize their contents, and store the resulting embeddings in a Pinecone database. It includes the following key features:

1.PDF Vectorization: Automatically vectorizes the content of uploaded PDFs using the integrated vectorization pipeline.

2.Pinecone DB Index Management: The app can check if the specified Pinecone index exists. If not, it will create a new one; otherwise, it will upsert the vectorized data to the existing index.

3.Streamlined Workflow: A simple and user-friendly Streamlit interface makes the process of managing Pinecone DB and vectorizing PDFs seamless.


## Authors

- [@yashsarnaik](https://www.github.com/yashsarnaik)


## Installation

pip install -r requirements.txt


    
## Run Locally

To deploy this project run

```bash
  streamlit run app.py
```


## Features

- Easy to use
- Supports both creating new index and upserting in existing one



# Hi, I'm Yash Sarnaik! 👋


## 🚀 About Me
I'm a Data Science and Gen AI Enthusiast.


## Tech Stack

Streamlit , Langchain, Pinecone Database, PyPDF2


