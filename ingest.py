import os
from pathlib import Path
from dotenv import load_dotenv
from uuid import uuid4
# -------------------- LOAD LANGCHAIN MODULES --------------------
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
  
# -------------------- LOAD ENV --------------------s

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION")   
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")       
INDEX_NAME = os.getenv("PINECONE_INDEX")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")



DATA_URLS = Path("data/urls.txt")
LOCAL_DOCS_DIR = Path("data/local_docs")

# -------------------- VALIDATE ENV --------------------
required_envs = [OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_REGION, PINECONE_CLOUD]
if not all(required_envs):
    raise SystemExit("Please set OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, and CLOUD in your .env")

# -------------------- LOAD DOCUMENTS --------------------
docs = []

# 1️ Load URLs
if DATA_URLS.exists():
    urls = [u.strip() for u in DATA_URLS.read_text().splitlines() if u.strip() and not u.startswith("#")]
    if urls:
        print(f" Loading {len(urls)} URLs...")
        loader = UnstructuredURLLoader(urls=urls)
        docs += loader.load()

# 2️ Load local PDFs
if LOCAL_DOCS_DIR.exists():
    print(f" Loading local documents from {LOCAL_DOCS_DIR}...")
    loader = DirectoryLoader(LOCAL_DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs += loader.load()

if not docs:
    raise SystemExit(" No documents found. Add URLs or local PDFs.")

print(f"Total documents loaded: {len(docs)}")

# -------------------- SPLIT DOCUMENTS --------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"Total chunks after splitting: {len(chunks)}")

# -------------------- CREATE EMBEDDINGS --------------------

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
)

# --------------------------
# Initialize Pinecone
# --------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, create if not
if not pc.has_index(INDEX_NAME):
    print(f" Creating Pinecone index '{INDEX_NAME}' ")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,       
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

index = pc.Index(INDEX_NAME)


# --------------------------
# Upload to Pinecone
# --------------------------
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

uuids = [str(uuid4()) for _ in range(len(chunks))]
print(f"Generated UUIDs: {uuids}")

# Add chunks to Pinecone
vector_store.add_documents(documents=chunks, ids=uuids)
print("✅ Documents successfully uploaded to Pinecone.")



llm = ChatOpenAI(
    model="gpt-4o-mini",   # or gpt-4o / gpt-3.5-turbo
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)


retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # fetch top 3 relevant chunks

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # optional, shows source docs
)


chat_history = []

print("Chat system ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain.invoke({
        "question": query,
        "chat_history": chat_history
    })

    answer = result["answer"]
    sources = result.get("source_documents", [])

    print(f"\n: {answer}\n")

    if sources:
        print("Sources:")
        for s in sources:
            print(" -", s.metadata.get("source", "Unknown"))

    # Update chat history
    chat_history.append((query, answer))
