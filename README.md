**AI Health Chatbot**

A Python-based AI Health Chatbot that allows you to ask questions about health-related documents (PDFs, medical articles, or trusted URLs) using OpenAI embeddings and Pinecone vector database. Built with LangChain, the chatbot supports context-aware conversations for multi-turn interactions.

** Features**

Load and process medical documents from local PDFs or URLs.
Split documents into manageable chunks for efficient retrieval.
Generate OpenAI embeddings for each chunk.
Store embeddings in Pinecone vector database for fast semantic search.
Context-aware chat using LangChain's ConversationalRetrievalChain.
Display source documents used for answering queries.
Multi-turn conversation support for follow-up health questions.


**Add your health documents:**
 PDFs in data/local_docs/
 URLs in data/urls.txt (one URL per line)
