
import json
import logging
import time
from typing import Any, List, Literal
import uuid
from dataclasses import dataclass

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import requests
from chromadb.api.models.Collection import Collection
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dataclasses import field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_settings import BaseSettings, SettingsConfigDict


collection = None
agent = None
conversations = {}  # Dict[session_id, List[ModelMessage]]
model = None

@dataclass
class Deps:
    collection: Collection
    thoughts: List[str] = field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_key: str = Field(alias="SYVAI_API_KEY")
    model_name: str = "syvai/danskgpt-v2.1"
    base_url: str = "https://api.syv.ai/v1"
    embedding_model_name: str = "intfloat/multilingual-e5-large"

settings = Settings()


# Init VectorStore
def build_vector_store():
    url = "https://raw.githubusercontent.com/syv-ai/it-center-fyn/main/datasets/teknologier.json"
    response = requests.get(url)
    data = response.json()
   
    client = chromadb.Client()

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model_name
    )
    
    try:
        client.delete_collection(name="knowledge_base") # Deleting collection for demo purposes
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name="knowledge_base",
        embedding_function=ef,
        metadata={"description": "Teknologier database"}
    )
   
    ids = []
    documents = []
    metadatas = []
    
    for item in data:
        doc_id = str(uuid.uuid4())
        ids.append(doc_id)
        documents.append(item["content"])
        metadatas.append({
            "title": item["title"],
            "content": item["content"],
            "source": item["source"]
        })
    print("Generating embeddings...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"Inserted {len(ids)} documents into vector store")
    return collection


async def retrieve(ctx: RunContext[Any], search_query: str) -> str:
    """Søg i vores database og returner relevante kilder.

Args:
    search_query: Brugerens spørgsmål.
"""

    results = ctx.deps.collection.query(
        query_texts=[search_query],
        n_results=3
    )
    return json.dumps(results['metadatas'][0], ensure_ascii=False, indent=2)


logger = logging.getLogger("llm_api")
logger.setLevel(logging.INFO)


async def reflect(ctx: RunContext[Deps], thought: str) -> str:
    """Analyser kilderne fra 'retrieve' og evaluer din forståelse.

    Vær analytisk og kritisk i din refleksion:
    - Hvilke kilder er mest relevante for spørgsmålet?
    - Besvarer kilderne spørgsmålet direkte eller kun delvist?
    - Er der modsigelser eller manglende information?
    - Hvordan strukturerer jeg svaret bedst for klarhed?

    Args:
        thought: Din detaljerede analyse og vurdering af kilderne

    Returns:
        Bekræftelse på din refleksion
    """
    ctx.deps.thoughts.append(thought)
    return f"[Tanke]{thought}[/Tanke]"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup
    global collection, model, agent

    try:
        # Default settings, can be changed at runtime
        provider = OpenAIProvider(base_url=settings.base_url, api_key=settings.api_key)
        model = OpenAIChatModel(model_name=settings.model_name, provider=provider)

        agent = Agent(
            model=model,
            tools=[retrieve, reflect],
            system_prompt="""Du er en dansk assistent.
ARBEJDSGANG - Følg disse trin i rækkefølge:
1. Brug 'retrieve' værktøjet til at søge i kilderne
2. Brug 'reflect' værktøjet til at tænke over spørgsmålet og kilderne, og planlægge dit svar
3. Giv dit endelige svar baseret på kilderne og din refleksion

Svar direkte, korrekt og naturligt på dansk.
Brug kun fakta fra kilderne.
Hvis du ikke kan finde svaret i kilderne, så sig dette tydeligt.
Inkluder kildehenvisninger i teksten som [1], [2], ... og tilføj en liste over kilder til sidst.
Tilføj en "Kilder:" sektion til sidst med format:
   Kilder:
   [1] <title>
   [2] <title>
   [3] <title>
""",)
        logger.info(f"Preloaded model '{settings.model_name}'")
    except Exception as e:
        logger.exception(f"Failed to preload model '{settings.model_name}': {e}")

    try:
        # Replace with your own vector store & embedding model
        collection = build_vector_store()
    except Exception as e:
        logger.exception(f"Failed to preload embedding model: {e}")

    yield
    # Shutdown (if needed)



app = FastAPI(title="OpenAI-Compatible LLM API", lifespan=lifespan)

# ---------------------------
# /v1/chat/completions – altid med kildehenvisning
# ---------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 500
    session_id: int

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model_name = request.model
    agent.model = model

    # Extract user message
    user_message = ""
    for msg in request.messages:
        if msg.role == "user":
            user_message += f"{msg.content.strip()}\n"
    
    # Use a session ID from request
    session_id = getattr(request, 'session_id', 'default')
    history = conversations.get(session_id, [])

    logger.info(f"Received chat request for model '{model_name}'")
    
    start_time = time.time()
    
    try:
        # Run agent with RAG
        deps = Deps(collection=collection)
        result = await agent.run(user_message, deps=deps, message_history=history)

        # Print thoughts
        if result and deps.thoughts:
            print(f"DEBUG: len(deps.thoughts) = {len(deps.thoughts)}")
            print(f"DEBUG: deps.thoughts = {deps.thoughts}")

        # Update conversation history
        conversations[session_id] = result.all_messages()
        
        response_text = result.output
        latency = time.time() - start_time
        finish_reason = result.all_messages()[-1].finish_reason
        usage = result.usage()
        
        logger.info(f"Generated chat completion: {response_text}")
        logger.info(f"Measuring latency: {latency:.3f} seconds")
    except Exception as e:
        logger.exception(f"Error during chat generation: {e}")
        return {"error": str(e)}

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)