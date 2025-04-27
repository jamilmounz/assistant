# LangManus API

A LangGraph-based agent workflow API with event streaming to the frontend.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
uvicorn src.api.app:app --reload --port 8000
```

## Testing the Streaming Functionality

1. Start the API server as described above

2. In a separate terminal, run the test script:
```bash
python test_stream.py
```

This will connect to the API and display all streamed events in the terminal.

## API Endpoints

### `/api/chat` (POST)

Accepts a JSON payload with the following structure:
```json
{
  "messages": [
    {"role": "user", "content": "Your query here"}
  ],
  "deep_thinking_mode": false,
  "search_before_planning": false
}
```

Returns a Server-Sent Events (SSE) stream with the following event types:
- `start_of_workflow`: Workflow initialization
- `start_of_agent`: Agent starts processing
- `end_of_agent`: Agent finishes processing
- `start_of_llm`: LLM starts generating
- `end_of_llm`: LLM finishes generating
- `message`: Content from LLM is streamed
- `tool_call`: Tool is being called
- `tool_call_result`: Result of a tool call
- `end_of_workflow`: Workflow completion

Each event includes relevant data according to the event-stream-protocol documentation.

### `/` (GET)

Health check endpoint returning status information. 