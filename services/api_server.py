from contextlib import asynccontextmanager

from fastapi import FastAPI

from orchestrator.langgraph_dag import initialize_orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    initialize_orchestrator()
    print("SERA initialized successfully.")
    yield
    # Shutdown logic (if needed)
    print("SERA shutting down...")


app = FastAPI(title="SERA API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "message": "SERA core initialized"}
