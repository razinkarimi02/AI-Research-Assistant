# AI Content Research Assistant

The **AI Content Research Assistant** is a modular, multi-step system designed to process user queries, retrieve relevant information, and generate precise answers using Retrieval-Augmented Generation (RAG) workflows. It integrates advanced retrieval techniques like BM25, semantic search, and reranking, along with large language models (LLMs) for generating high-quality responses.

---

## 🚀 Features

- **Multi-Step Workflow**: Orchestrates tasks like internet search, document retrieval, and answer generation.
- **Retrieval-Augmented Generation (RAG)**: Combines BM25 and semantic search for document retrieval and reranking.
- **LLM Integration**: Supports models like `llama3:8b` and `gpt-4o-mini` for generating responses.
- **Streamlit UI**: Provides an interactive interface for querying the system.
- **Extensible Design**: Modular architecture for easy customization and scalability.

---

## 📂 Project Structure

```
ai-research-assistant/
│
├── README.md                     # Project overview and instructions
├── pyproject.toml / requirements.txt  # Dependencies
├── .env                          # Environment variables
├── .gitignore                    # Git ignore rules
│
├── src/
│   ├── agents.py                 # Agent logic
│   ├── mcp_search_client.py      # MCP search client
│   ├── mcp_tavily_server.py      # MCP server logic
│   ├── workflow.py               # Workflow logic
│
├── utils/
│   ├── logger.py                 # Logging utilities
│
├── assets/
│   └── workflow.png              # Workflow diagram
```

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ai-research-assistant.git
   cd ai-research-assistant
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory.
   - Add the following variables:
     ```
     PORT=8000
     URL_SCHEMA="http"
     LOGGER_ROOT="./logs"
     ```

---

## 🚀 Usage

### **Run the API**
1. Start the Flask/FastAPI server:
   ```bash
   python apps/api/main.py
   ```

2. Access the API at `http://localhost:8000`.

---

### **Run the Streamlit App**
1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open the app in your browser at `http://localhost:8501`.

---

## 🧠 Workflow Overview

1. **Orchestrator**:
   - Decides the next action: `research`, `rag`, or `end`.
   - Uses LLMs to analyze the query and available information.

2. **Research**:
   - Performs internet searches using `mcp_internet_search`.

3. **RAG (Retrieval-Augmented Generation)**:
   - Retrieves relevant documents using BM25 and semantic search.
   - Combines and reranks results using a cross-encoder.

4. **Final Answer**:
   - Generates a concise answer using an LLM.

---

## 🧪 Testing

Run the test suite using `pytest`:
```bash
pytest tests/
```

---

## 📈 Future Enhancements

- Add support for additional LLMs.
- Implement caching for frequently asked queries.
- Improve error handling and logging.
- Add more nodes for specialized tasks (e.g., summarization, translation).

---

## 🤝 Contributing

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [BM25](https://github.com/dorianbrown/rank_bm25)
- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.ai/)

---

Feel free to customize this `README.md` further to suit your project! Let me know if you'd like to add or modify anything.

