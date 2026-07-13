import os
import json
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI

app = FastAPI(title="GraphRAG Pipeline Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------
# Configuration & Client Initialization
# ----------------------------------------------------------------

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
if not AIPIPE_TOKEN:
    raise RuntimeError("AIPIPE_TOKEN environment variable not set.")

# FIX: Restored the exact working base_url proxy endpoint from your invoice parser
client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url="https://aipipe.org/openrouter/v1"
)

# Using the working Gemini model tag
MODEL_NAME = "google/gemini-2.5-flash"

# ----------------------------------------------------------------
# Request Data Validation Models
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

def ask_graphrag_llm(prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise backend parsing data engine. You must output raw, valid JSON only. Do not wrap code in markdown blocks."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )
        
        raw_content = response.choices[0].message.content.strip()
        
        # Clean up text if the model appends decorative markdown wrappers
        if raw_content.startswith("```"):
            if "\n" in raw_content:
                raw_content = raw_content.split("\n", 1)[1]
            else:
                raw_content = raw_content.lstrip("`json").lstrip("`")
        if raw_content.endswith("```"):
            raw_content = raw_content.rsplit("```", 1)[0]
            
        return json.loads(raw_content.strip())
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Automated LLM pipeline processing error: {str(e)}"
        )


# ----------------------------------------------------------------
# Status Endpoints
# ----------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "running", "service": "GraphRAG Pipeline"}


@app.get("/health")
def health():
    return {"ok": True}


# ----------------------------------------------------------------
# Endpoint 1: POST /extract-graph
# ----------------------------------------------------------------

@app.post("/extract-graph")
async def extract_graph(data: ExtractRequest):
    prompt = f"""
Extract entities and relationships from the text chunk below.

Allowed Entity Types: [Person, Organization, Product, Framework]
Allowed Relationship Types: [CREATED, FOUNDED, DEVELOPED, INTEGRATED_INTO, HIRED, AUTHORED]

Text:
{data.text}

Return validation data ONLY inside this explicit JSON target schema format:
{{
  "entities": [
    {{"name": "Entity Name", "type": "Type"}}
  ],
  "relationships": [
    {{"source": "Source Name", "target": "Target Name", "relation": "RELATION_TYPE"}}
  ]
}}
"""
    result = ask_graphrag_llm(prompt)
    return {
        "entities": result.get("entities", []),
        "relationships": result.get("relationships", [])
    }


# ----------------------------------------------------------------
# Endpoint 2: POST /graph-query
# ----------------------------------------------------------------

@app.post("/graph-query")
async def graph_query(data: GraphQueryRequest):
    entities_block = "\n".join(
        f"- {e['name']} ({e['type']})" for e in data.graph.get("entities", [])
    )
    relationships_block = "\n".join(
        f"- {r['source']} -> {r['relation']} -> {r['target']}" for r in data.graph.get("relationships", [])
    )

    prompt = f"""
Perform step-by-step navigation across the graph data map below to solve the given question.

Knowledge Graph Context:
Entities:
{entities_block}

Relationships:
{relationships_block}

User Question:
{data.question}

If the question cannot be safely answered from the facts above, return exactly:
{{
  "answer": "Unknown",
  "reasoning_path": [],
  "hops": 0
}}

Otherwise, list every node name traversed in sequence inside reasoning_path and return:
{{
  "answer": "Final extracted answer node string",
  "reasoning_path": ["First Node", "Second Node", "Third Node"],
  "hops": 2
}}
"""
    result = ask_graphrag_llm(prompt)
    return {
        "answer": result.get("answer", "Unknown"),
        "reasoning_path": result.get("reasoning_path", []),
        "hops": int(result.get("hops", 0))
    }


# ----------------------------------------------------------------
# Endpoint 3: POST /community-summary
# ----------------------------------------------------------------

@app.post("/community-summary")
async def community_summary(data: CommunitySummaryRequest):
    relationships_block = "\n".join(
        f"- {r.get('source')} -> {r.get('relation')} -> {r.get('target')}" for r in data.relationships
    )

    prompt = f"""
Generate a clear summary outlining the thematic connections inside this sub-community group.

Entities:
{', '.join(data.entities)}

Relationships Map:
{relationships_block}

Return exactly in this JSON format structure:
{{
  "summary": "Detailed context summary text explaining the structural entity relationships."
}}
"""
    result = ask_graphrag_llm(prompt)
    return {
        "community_id": data.community_id,
        "summary": result.get("summary", "")
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
