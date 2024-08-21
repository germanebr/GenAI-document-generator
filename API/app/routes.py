import logging

from typing import List, Optional
from flask import request

from app import app
from app.gemini import Gemini

@app.route("/api/health")
def home():
    logging.info("Home route accessed")
    return ({"message": "connected successfully"})

@app.route("/test_llm")
def test_llm():
    llm = Gemini()
    query = "Say that the LLM is operational"

    response = llm.generate_draft(query = query)
    return {"status": True,
            "draft": response}

@app.route("/get_draft", methods=['POST'])
def generate_draft():
    query: str = request.json['query']
    paths: Optional[List] = request.json['paths']
    llm = Gemini()

    if not paths:
        response = llm.generate_draft(query = query)
    
    response = llm.generate_draft(query = query,
                                  paths = paths)
    return {"status": True,
            "draft": response}