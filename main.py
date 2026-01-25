import logging
import os
import uvicorn
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from waitress import serve
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import warnings
import logging
load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)


PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("URL_SCHEMA", "0.0.0.0")

from utils.logger import create_logger
from src.workflow import run_workflow


logging_root = os.getenv("LOGGER_ROOT")
logger = create_logger(logging_root, "MAIN")
app = Flask(__name__)


# ------------ Request Schema ------------
class QueryRequest(BaseModel):
    query: str
    max_depth: int = 3


# ------------ Health Check ------------
@app.get("/")
def health_check():
    return {"status": "running", "app": "AI Content Research Assistant"}


# ------------ Main Endpoint ------------
@app.post("/query")
async def process_query():
    """
    Receives user query → sends it to LangGraph workflow → returns output.
    """
    # try:
    if True:
        request_id = uuid.uuid4()
        files = request.files.getlist('files')
        print("Files received: ", files)
        query= request.form.get("query")

        if query is None:
            return jsonify({"error": "Invalid or missing query"}), 400

        logger.info("Workflow has started for query: %s, Request ID: %s", query, request_id)
        result =await run_workflow(query, files,logger)
        return jsonify(
            content={"query": query, "result": result},
            status_code=200
        )
    
    # except Exception as e:
    #     return JSONResponse(
    #         content={"error": str(e)},
    #         status_code=500
    #     )


if __name__ == "__main__":    
    serve(app, host='0.0.0.0', port=PORT,  url_scheme='https')
