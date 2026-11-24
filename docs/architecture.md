# SERA — Architecture

This document separates the implemented ingestion baseline from the approved research-agent target. The governing specification is [Design Document & Execution Plan v3.0](design_doc.md), particularly §§3–6 and §13. The diagram below is the approved target, not a claim that its planned components already exist.

## Current implementation

```text
CLI (query or paper ID)
    → arXiv retrieval / paper metadata
    → PDF download → page parsing → token chunking → embeddings
    → vector adapter (Chroma or JSON) and SQLite metadata / vector links
```

The ingestion pipeline is dependency-injected and uses single-process asyncio with bounded download, parse and embedding concurrency. The FastAPI application exposes a health endpoint; its LangGraph initialization is a placeholder, not a working research graph.

| Area | Current source | Status at baseline |
| --- | --- | --- |
| arXiv retrieval | `agents/retriever_arxiv_real.py` | Implemented retrieval client; global rate limiter required |
| PDF download, parse, chunk | `services/pdf_downloader.py`, `parsers/pdf_parser.py`, `utils/chunker.py` | Implemented |
| Ingestion and embeddings | `agents/researcher_agent.py`, `embeddings/embedder.py` | Implemented; F1–F5 repairs required |
| Metadata and vector adapters | `memory/metadata_db.py`, `memory/chroma_adapter.py`, `memory/simple_adapter.py` | Implemented; corpus integrity and durable linkage not established |
| API and orchestrator | `services/api_server.py`, `orchestrator/langgraph_dag.py` | Health endpoint implemented; research orchestration planned |

The corpus currently lacks full stored chunk text, reliable interrupted-ingestion recovery, fail-closed embedding behavior and enforced embedding provenance. The legacy OpenAI path also requires repair or removal. See [README limitations](../README.md#known-foundation-limitations) and §13 Phase 0.3 of the design. SQLite per-operation transactions do not establish atomicity across SQLite and the vector store.

## Approved target architecture

The following legend and component diagram are reproduced from design v3.0 §5.1. The labels refer to the approved pre-build baseline: `[B]` for existing functionality (not necessarily an exposed agent tool), `[R]` for existing functionality needing repair, and `[N]` for new functionality.

`[B]` = built today · `[R]` = built, needs a Phase 0 repair · `[N]` = new

```
┌────────────────────────────────────────────────────────────────┐
│  CLIENT: CLI · REST · SSE                                        │
└──────────────────────────────┬─────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────┐
│  API (FastAPI)                                   [B health][N]   │
│  POST /research → run_id · GET /runs/{id} · GET /runs/{id}/events│
│  POST /runs/{id}/resume                                          │
└──────────────────────────────┬─────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────┐
│  ORCHESTRATOR (LangGraph)                                  [N]  │
│                                                                  │
│   PLANNER ──▶ EXECUTOR ──▶ CRITIC ──▶ MEMORY WRITE ──▶ (loop)   │
│   (LLM)       (code)       (rules+LLM)                           │
│      ▲                       │ replan                            │
│      └───────────────────────┘                                   │
│                    ──▶ SYNTHESIZER (LLM) ──▶ FINAL CHECK         │
│                                                                  │
│   AgentState checkpointed after every node                       │
└───────┬─────────────────────┬─────────────────────┬────────────┘
        │                     │                     │
┌───────▼────────┐  ┌─────────▼──────────┐  ┌───────▼───────────┐
│ MEMORY         │  │ TOOLS              │  │ LLM GATEWAY   [N] │
│ context    [N] │  │ corpus_search  [N] │  │ role→model map    │
│ episodic   [N] │  │ arxiv_search   [B] │  │ structured output │
│ semantic   [N] │  │ ingest_paper   [R] │  │ retries/timeouts  │
│ corpus     [R] │  │ code_exec      [N] │  │ prompt caching    │
└───────┬────────┘  └─────────┬──────────┘  │ cost ledger       │
        │                     │             └───────┬───────────┘
        │       ┌─────────────▼─────────────┐       │
        │       │ INGESTION PIPELINE    [R] │       │
        │       │ retrieve→download→parse→  │       │
        │       │ chunk→embed→store→link    │       │
        │       └───────────────────────────┘       │
┌───────▼───────────────────────────────────────────▼────────────┐
│  OBSERVABILITY (runs.db: runs · node_events · tool_calls ·     │
│  critic_scores · llm_calls) + static dashboard             [N] │
└──────────────────────────────────────────────────────────────────┘
```

## Architecture boundaries

The planned agent uses an LLM planner, a code executor, a rules-first critic with LLM escalation, memory writes, synthesis and final checks. A reflector is reserved for Milestone 4. These are target roles, not implemented agents.

The target separates corpus and episodic vector collections, semantic memory, run traces, checkpoints and the committed corpus manifest as specified in §5.2. These target paths and schemas are not claims about the current storage layout.

Phase 0.3 must exit before agent-layer implementation. Follow the phase order, contracts and evaluation gates in the approved design. The former sketches' alternative vector backend, multi-agent layout, monitoring stack and interactive UI are not commitments of v3.0; the approved scope uses the diagram above and its specified observability. Multi-node execution and a chat UI are non-goals (§3.3).

The term self-improvement is conditional on the measured Milestone 4 result, not on the presence of a reflection component.
