import logging
from src.graph import build_graph
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import re

logger = logging.getLogger(__name__)
graph = build_graph()

app = FastAPI(
    title="LangManus API",
    description="API for LangManus LangGraph-based agent workflow",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def clean_response(text: str) -> str:
    """Clean the response format from the agent's messages."""
    # Remove the "Response from agent:" prefix
    text = re.sub(r'^Response from [^:]+:\s*', '', text)
    # Remove the <response> tags and the "Please execute the next step" instruction
    text = re.sub(r'<response>(.*?)</response>\s*\*Please execute the next step\.\*', r'\1', text, flags=re.DOTALL)
    return text.strip()

async def stream_workflow_response(user_messages: list,
                                   deep_thinking: bool = False,
                                   search_before_planning: bool = False):

    target_nodes = {"researcher", "coder", "browser",
                    "planner", "coordinator", "reporter", "supervisor"}

    async for delta in graph.astream(
        {
            "TEAM_MEMBERS": list(target_nodes),
            "messages": user_messages,
            "deep_thinking_mode": deep_thinking,
            "search_before_planning": search_before_planning,
        },
        stream_mode="updates",
    ):
        # Skip if delta is empty
        if not delta:
            continue
            
        node, patch = next(iter(delta.items()))          # one node finished
        
        # Skip if patch is None or doesn't contain messages
        if not patch or "messages" not in patch:
            continue                                      # nothing to show

        final_msg = patch["messages"][-1].content

        # Special-case the planner so the front-end can render the workflow UI
        if node == "planner":
            payload = {
                "node": "planner",
                "kind": "plan",
                "raw":   final_msg           # JSON string your planner wrote
            }
        else:
            payload = {
                "node": node,                # e.g. researcher / coordinator
                "kind": "message",
                "text": clean_response(final_msg)
            }

        #  send **JSON** so the client can just `JSON.parse()`
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

@app.post("/api/chat")
async def run_workflow(request: Request):
    body = await request.json()            # <-- read JSON once
    return StreamingResponse(
        stream_workflow_response(
            body["messages"],
            deep_thinking=body.get("deep_thinking_mode", False),
            search_before_planning=body.get("search_before_planning", False),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
