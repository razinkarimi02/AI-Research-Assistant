import streamlit as st
import logging
from workflow import run_workflow

# --------------------------------------------------
# Logger (simple)
# --------------------------------------------------
logger = logging.getLogger("rag-agent")
logger.setLevel(logging.INFO)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="AI MCP Chatbot", layout="wide")
st.title("🔍 AI MCP Chatbot")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# File upload
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload documents (PDF, TXT, DOCX)",
    accept_multiple_files=True
)

# Save uploaded files to temp paths
file_paths = []
if uploaded_files:
    for f in uploaded_files:
        path = f"/tmp/{f.name}"
        with open(path, "wb") as out:
            out.write(f.read())
        file_paths.append(path)

# --------------------------------------------------
# Render previous chat
# --------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# User input
# --------------------------------------------------
user_query = st.chat_input("Ask your question")

if user_query:
    # Show user message
    st.session_state.chat_history.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    # Spinner while processing
    with st.spinner("Thinking..."):
        try:
            answer = run_workflow(
                query=user_query,
                files=file_paths,
                logger=logger
            )
        except Exception as e:
            answer = f"❌ Error: {str(e)}"

    # Show assistant response
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
