import os
import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

app = FastAPI()

# Render automatically handles the PORT, and we load the AI Pipe Token securely
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN") 
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
MODEL_NAME = "openai/gpt-4.1-nano" 

# ----------------------------------------------------------------
# Validation Schemas
# ----------------------------------------------------------------
class ExtractRequest(BaseModel):
    chunk_id: str
    text: str

class GraphQueryRequest(BaseModel):
    question: str
    graph: Dict[str, Any]

class CommunitySummaryRequest(BaseModel):
    community_id: str
    entities: List[str]
    relationships: List[Any]

# ----------------------------------------------------------------
# Helper to call AI Pipe API
# ----------------------------------------------------------------
def call_aipipe_llm(prompt: str) -> dict:
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": { "type": "json_object" }
    }
    
    response = requests.post(AIPIPE_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    result_text = response.json()["choices"][0]["message"]["content"]
    return json.loads(result_text)

# ----------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------
@app.post("/extract-graph")
async def extract_graph(data: ExtractRequest):
    prompt = f"""
    Analyze this text and extract entities and relationships.
    Allowed Entity Types: Person, Organization, Product, Framework
    Allowed Relationship Types: FOUNDED, DEVELOPED, INTEGRATED_INTO, HIRED, AUTHORED, CREATED
    
    Text: "{data.text}"
    
    Respond strictly with this JSON layout:
    {{
      "entities": [ {{"name": "...", "type": "..."}} ],
      "relationships": [ {{"source": "...", "target": "...", "relation": "..."}} ]
    }}
    """
    return call_aipipe_llm(prompt)

@app.post("/graph-query")
async def graph_query(data: GraphQueryRequest):
    entities_str = ", ".join([f"{e['name']} ({e['type']})" for e in data.graph.get('entities', [])])
    relations_str = "\n".join([f"- {r['source']} -> {r['relation']} -> {r['target']}" for r in data.graph.get('relationships', [])])
    
    prompt = f"""
    Answer the user question using ONLY the provided knowledge graph data.
    
    Entities: {entities_str}
    Relationships:
    {relations_str}
    
    Question: {data.question}
    
    Respond strictly with this JSON layout:
    {{
      "answer": "String answer",
      "reasoning_path": ["EntityA", "EntityB"],
      "hops": 2
    }}
    """
    return call_aipipe_llm(prompt)

@app.post("/community-summary")
async def community_summary(data: CommunitySummaryRequest):
    relations_str = "\n".join([f"- {r.get('source')} -> {r.get('relation')} -> {r.get('target')}" for r in data.relationships])
    
    prompt = f"""
    Summarize this specific sub-community of entities and relationships into one clean sentence.
    
    Entities: {', '.join(data.entities)}
    Relationships:
    {relations_str}
    
    Respond strictly with this JSON layout:
    {{
      "summary": "Your clear summary sentence goes here."
    }}
    """
    llm_output = call_aipipe_llm(prompt)
    return {
        "community_id": data.community_id,
        "summary": llm_output.get("summary", "")
    }

if __name__ == "__main__":
    # Render binds dynamic port assignments to the PORT env variable automatically
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)