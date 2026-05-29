# 🎙️ EchoNote — Talk to Your Documents using RAG

EchoNote is a **voice-enabled AI document assistant** that lets users upload PDF files, ask natural language questions, and receive **accurate, document-grounded answers** in both **text and voice**.

Built using **Retrieval-Augmented Generation (RAG)**, VoxDoc combines semantic search, large language models, and speech synthesis to transform static documents into interactive knowledge sources.

---

## ✨ Key Highlights

- 📄 Upload and analyze PDF documents  
- 🔍 Semantic search powered by vector embeddings  
- 🧠 Retrieval-Augmented Generation (RAG) architecture  
- 💬 Multiple chat sessions with independent memory  
- 🎙️ Voice responses via text-to-speech  
- ⚡ High-speed inference using Groq (LLaMA 3)  
- 🗄️ Local vector storage using Qdrant  

---

## 🧩 How EchoNote Works

1. A PDF document is uploaded by the user  
2. The document is split into meaningful text chunks  
3. Each chunk is converted into a vector embedding  
4. Embeddings are stored in a Qdrant vector database  
5. User asks a question in natural language  
6. Relevant chunks are retrieved based on semantic similarity  
7. The LLM generates a context-aware response  
8. The response is displayed and spoken aloud  

---

## 🏗️ Architecture Overview

PDF Upload
↓
Text Chunking
↓
Embedding Generation
↓
Qdrant Vector Database
↓
User Query → Embedding
↓
Relevant Context Retrieval
↓
LLM (Groq - LLaMA 3)
↓
Text Output + Voice Response

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|-----------|
| Frontend | Streamlit |
| LLM | Groq (LLaMA-3.3-70B) |
| Embeddings | FastEmbed |
| Vector Database | Qdrant |
| Document Loader | LangChain |
| Text Chunking | RecursiveCharacterTextSplitter |
| Voice Output | gTTS |
| Language | Python |

---

## 📁 Project Structure

voxdoc/
│
├── chatbot.py # Streamlit UI & RAG pipeline
├── agents.py # LLM agent and execution logic
├── requirements.txt # Project dependencies
├── .env # API keys and environment variables
└── qdrant_db/ # Local vector database (auto-generated)

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/EchoNote.git
cd voxdoc
2️⃣ Create a Virtual Environment (Recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt

5️⃣ Run VoxDoc
bash
Copy code
streamlit run chatbot.py

🧪 How to Use
Click Initialize System

Upload a PDF document

Ask questions related to the document

Switch between multiple chat sessions

Listen to AI-generated voice answers

📄 License
This project is licensed under the MIT License.
