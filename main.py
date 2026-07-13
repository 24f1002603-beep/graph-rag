import os
import json
import requests
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="GraphRAG Pipeline")

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

if not AIPIPE_TOKEN:
    raise RuntimeError("AIPIPE_TOKEN environment variable not set.")

AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
MODEL_NAME = "openai/gpt-4.1-nano"

# ----------------------------------------------------------------
# Request Models
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
# Helper Function
# ----------------------------------------------------------------

def call_aipipe_llm(prompt: str) -> dict:
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {
            "type": "json_object"
        }
    }

    try:
        response = requests.post(
            AIPIPE_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        content = result["choices"][0]["message"]["content"]

        return json.loads(content)

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"AIPipe request failed: {str(e)}"
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="LLM returned invalid JSON."
        )


# ----------------------------------------------------------------
# Health Endpoints
# ----------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "GraphRAG Pipeline"
    }


@app.get("/health")
def health():
    return {
        "ok": True
    }


# ----------------------------------------------------------------
# Endpoint 1
# ----------------------------------------------------------------

@app.post("/extract-graph")
async def extract_graph(data: ExtractRequest):

    prompt = f"""
Extract entities and relationships from the following text.

Allowed Entity Types:
- Person
- Organization
- Product
- Framework

Allowed Relationship Types:
- CREATED
- FOUNDED
- DEVELOPED
- INTEGRATED_INTO
- HIRED
- AUTHORED

Return ONLY valid JSON.

Text:
{data.text}

JSON format:

{{
  "entities":[
    {{
      "name":"",
      "type":""
    }}
  ],
  "relationships":[
    {{
      "source":"",
      "target":"",
      "relation":""
    }}
  ]
}}
"""

    result = call_aipipe_llm(prompt)

    return {
        "entities": result.get("entities", []),
        "relationships": result.get("relationships", [])
    }


# ----------------------------------------------------------------
# Endpoint 2
# ----------------------------------------------------------------

@app.post("/graph-query")
async def graph_query(data: GraphQueryRequest):

    entities = "\n".join(
        f"{e['name']} ({e['type']})"
        for e in data.graph.get("entities", [])
    )

    relationships = "\n".join(
        f"{r['source']} -> {r['relation']} -> {r['target']}"
        for r in data.graph.get("relationships", [])
    )

    prompt = f"""
Answer the question ONLY using the supplied knowledge graph.

Entities:

{entities}

Relationships:

{relationships}

Question:

{data.question}

If the answer cannot be inferred from the graph, return

{{
  "answer":"Unknown",
  "reasoning_path":[],
  "hops":0
}}

Otherwise return ONLY JSON in this format

{{
  "answer":"",
  "reasoning_path":["","",""],
  "hops":2
}}
"""

    result = call_aipipe_llm(prompt)

    return {
        "answer": result.get("answer", "Unknown"),
        "reasoning_path": result.get("reasoning_path", []),
        "hops": result.get("hops", 0)
    }


# ----------------------------------------------------------------
# Endpoint 3
# ----------------------------------------------------------------

@app.post("/community-summary")
async def community_summary(data: CommunitySummaryRequest):

    relationships = "\n".join(
        f"{r.get('source')} -> {r.get('relation')} -> {r.get('target')}"
        for r in data.relationships
    )

    prompt = f"""
Summarize the following graph community.

Entities:
{", ".join(data.entities)}

Relationships:
{relationships}

Return ONLY valid JSON.

{{
  "summary":""
}}
"""

    result = call_aipipe_llm(prompt)

    return {
        "community_id": data.community_id,
        "summary": result.get("summary", "")
    }


# ----------------------------------------------------------------
# Run Server
# ----------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
