from flask import Flask, render_template, request, session
from src.helpers import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.websearch import search_pubmed
from dotenv import load_dotenv
from src.prompt import *
import os
import sqlite3
import uuid
from pathlib import Path

app = Flask(__name__)
app.secret_key = "development-secret-key"

load_dotenv(Path(__file__).resolve().parents[2] / "secret" / ".env")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATABASE = Path(__file__).with_name("chat_history.db")

def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection

def init_db():
    with get_db() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)

def get_user_id():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]

def save_message(user_id, role, content):
    with get_db() as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (user_id, role, content)
            VALUES (?, ?, ?)
            """,
            (user_id, role, content)
        )

def get_messages_for_user(user_id):
    with get_db() as connection:
        return connection.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY id
            """,
            (user_id,)
        ).fetchall()

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

chatModel = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("human", "{input}"),
])

router = router_prompt | chatModel

def route(msg):
    return router.invoke({"input": msg}).content.strip().lower()

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

def retrieve(query, k=3,threshold=0.35):
        hits = docsearch.similarity_search_with_score(query, k=k)
        docs =[doc for doc, score in hits if score >= threshold]
        for doc in docs:
            doc.metadata["source_type"] = "pdf"
        return docs

def get_context(msg):
    ziel = route(msg)
    if ziel == "pubmed":
        return search_pubmed(msg)
    if ziel == "pdf":
        return retrieve(msg)
    return []

@app.route("/")
def index():
    user_id = get_user_id()
    messages = get_messages_for_user(user_id)
    return render_template("chat.html", messages=messages)

@app.route("/get", methods=["POST"])
def chat():
    user_id = get_user_id()
    msg = request.form["msg"]

    documents = get_context(msg)

    reference = set()
    for doc in documents:
        if doc.metadata.get("source_type") == "pubmed":
            reference.add(f"{doc.metadata['title']} — {doc.metadata['url']}")
        else:
            name = Path(doc.metadata.get("source", "")).name
            page = int(doc.metadata.get("page", 0)) + 1
            reference.add(f"{name} (Page: {page})")     
    answer = question_answer_chain.invoke({"input": msg, "context": documents})
    if reference:
        answer = answer + "\n\nReferences:\n" + "\n".join(reference)
    
    save_message(user_id, "user", msg)
    save_message(user_id, "assistant", answer)

    return answer


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=8080, debug=True)