                    User
                      │
                      ▼
               Gradio Frontend
                      │
                      ▼
              chat_engine()
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   SQLite Memory  Router LLM   Session Manager
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
    General LLM     Web Search      RAG Search
      (Gemini)     (DuckDuckGo)      (FAISS)
         │              │              │
         └──────────────┴──────────────┘
                        │
                        ▼
                    Gemini API
                        │
                        ▼
                 Stream Response
                        │
                        ▼
             Save into SQLite Memory
                        │
                        ▼
                  Display in Gradio
