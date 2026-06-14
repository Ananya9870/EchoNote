# 📚 EchoNote – AI-Powered Document Learning Assistant

EchoNote is an AI-powered document tutor that transforms PDFs, Word documents, PowerPoint presentations, notes, and images into an interactive learning experience. Instead of simply answering questions, EchoNote teaches concepts, generates exam-oriented questions, explains formulas step-by-step, summarizes content, and provides voice based responses.

## 🚀 Features

* 📄 Upload PDF, DOCX, PPTX, TXT, Markdown, and Image files
* 🔍 RAG-based document understanding using Qdrant Vector Database
* 🤖 AI-powered tutoring and concept explanation
* 📝 Automatic generation of important exam questions
* 📊 Formula and numerical problem solving with step-by-step explanations
* 🖼️ OCR support for extracting text from images
* 🎙️ Text-to-Speech voice responses
* 💬 Multi-chat support with persistent chat history
* 📚 Document-specific memory and retrieval
* 🔒 Chat session management with memory cleanup

## 🏗️ Tech Stack

### Frontend

* Streamlit

### Backend & AI

* Python
* OpenAI Agents SDK
* FastEmbed
* Qdrant Vector Database

### Document Processing

* PyMuPDF
* Python-Docx
* Python-PPTX
* OCR (Tesseract)

### Voice

* gTTS (Google Text-to-Speech)

## ⚙️ System Architecture

1. User uploads a document.
2. Document content is extracted.
3. Text is split into semantic chunks.
4. Embeddings are generated using FastEmbed.
5. Chunks are stored in Qdrant Vector Database.
6. User asks questions.
7. Relevant context is retrieved through semantic search.
8. AI Agent generates context-aware responses.
9. Responses can be converted into speech.

## 📂 Project Structure

```bash
EchoNote/
├── chatbot.py
├── agents.py
├── requirements.txt
├── speeches/
├── qdrant_db/
├── chat_history.json
└── README.md
```

## 🛠️ Installation

```bash
git clone https://github.com/Ananya9870/EchoNote.git
cd EchoNote

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## ▶️ Run Locally

```bash
streamlit run chatbot.py
```

## 💡 Example Use Cases

* Learn from college notes
* Prepare for examinations
* Generate important questions
* Understand formulas and derivations
* Convert study material into audio explanations
* Interact with research papers through natural language

## 🔮 Future Enhancements

* Multi language tutoring
* Interactive quizzes and assessments
* YouTube video learning integration
* Diagram and graph interpretation
* Personalized learning paths
* Export notes and summaries

## 👩‍💻 Author

Ananya Kriti

Software Engineer | AI/ML Developer | RAG & Generative AI Enthusiast

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.
