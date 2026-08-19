from dotenv import load_dotenv
import os
from pathlib import Path
from src.helpers import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

extracted_docs = load_pdf_file(data=str(DATA_DIR))
minimal_docs = filter_to_minimal_docs(extracted_docs)
text_chunks = text_split(minimal_docs)
embeddings = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embeddings
        metric="cosine",  # Similarity metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,)


docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name,)
