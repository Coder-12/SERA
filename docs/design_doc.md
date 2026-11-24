# SERA — Research Agent
## Design Document & Execution Plan (v3.0 — final, pre-build)

| | |
|---|---|
| **Owner** | Aklesh Mishra |
| **Repo** | github.com/Coder-12/SERA |
| **Baseline commit** | `3149561` (all findings below were verified against this commit) |
| **Date** | 11 October 2026 |
| **Status** | Ingestion layer built; agent layer not started |
| **Commit this file as** | `docs/design_doc.md` (replaces the empty `docs/design_notes.md`) |

---

## 0. How to read this document

| If you want… | Read |
|---|---|
| What exists today, honestly | §1 |
| What to fix before anything else | §2, then Phase 0 in §13 |
| What we are building and what we are not | §3 |
| The decisions that shape everything | §4 |
| How the system works | §5–§9 |
| How we know it works | §10 |
| What order to build in | §13 |
| What to paste into another model when stuck | §17 |

**Rules that override everything else in this document:**

1. **No claim without code.** Nothing goes into the README, a diagram, or the resume unless it can be verified from the repo in under a minute.
2. **No phase is complete without a logged eval run** in `eval/results/` (from Phase 2 onward).
3. **No change is an improvement unless it beats the noise floor** (§10.6).
4. **The sealed test set is opened exactly once** (§10.2).

---

## 1. Current state (verified at `3149561`)

### 1.1 What exists and works

| Component | File | Assessment |
|---|---|---|
| arXiv retrieval | `agents/retriever_arxiv_real.py` | Async Atom API client with retries. Good. |
| PDF download | `services/pdf_downloader.py` | Streaming, size caps, PDF-header check, `.part` file + atomic rename. Good. |
| PDF parsing | `parsers/pdf_parser.py` | PyMuPDF page extraction. Fine. |
| Chunking | `utils/chunker.py` | Token-sized overlapping chunks. Fine. |
| Embedding | `embeddings/embedder.py` | Sentence Transformers backend works. **Two serious defects — §1.3.** |
| Vector store interface | `memory/vector_store_interface.py` | Typed `Protocol` with error hierarchy. Good. |
| Vector backends | `memory/chroma_adapter.py`, `memory/simple_adapter.py` | Complete, tested. |
| Metadata DB | `memory/metadata_db.py` | SQLite in WAL mode, a fresh connection per operation, explicit `BEGIN`/`COMMIT`/`ROLLBACK` around every write, foreign keys on. **This is correct transactional design.** |
| Ingestion orchestrator | `agents/researcher_agent.py` | Dependency injection, bounded semaphores per stage, per-paper failure isolation. Good structure. **One correctness bug — §1.3.** |
| Tests | `tests/` | 18 modules. CI runs 11 of them (37 tests); the other 7 need network or model downloads. **Green when run locally.** |
| CI | `.github/workflows/ci.yml` | black + isort + hermetic pytest. Ruff and mypy are commented out. |

### 1.2 What does not exist

- **No LLM.** No generation call anywhere in the repo. The only OpenAI call is for embeddings.
- **No query path.** `similarity_search` is called only from tests. SERA can ingest papers; it cannot answer a question.
- **No agent layer.** `critic_agent.py`, `reader_agent.py`, `reflection_agent.py`, `synthesizer_agent.py` are 0 bytes. `orchestrator/langgraph_dag.py` is 3 lines that print a string.
- **No eval, episodic memory, observability, or self-improvement.** `eval/metrics.py` and `memory/episodic_db.py` are 0 bytes.
- `ResearcherAgent` is an ingestion pipeline, not an agent. The name is misleading.

### 1.3 Defects that block the agent layer

These were found by reading the code and confirmed by running it. Each one would silently corrupt retrieval, and retrieval quality is what every eval number depends on. **All five are fixed in Phase 0 before any agent code is written.**

| ID | Defect | Evidence | Consequence |
|---|---|---|---|
| **F1** | **Full chunk text is never stored.** Only a 200-character `text_preview` reaches the vector-store metadata and `metadata_db`. The full text is thrown away after embedding. | `researcher_agent.py` lines ~470–480 build vector metadata with `text_preview` only | A search can return 200-character snippets and nothing more. The synthesizer cannot ground claims, and citation accuracy cannot be measured. |
| **F2** | **Idempotency is poisoned by partial failure.** The paper row is written *before* embedding. The skip check only asks whether a row exists. | Proven: `tests/test_ingest_crash_recovery.py` (provided alongside this document) is red. Crash mid-ingest → healthy retry reports `success=True, n_vectors=0`. | Any paper whose ingestion was interrupted is permanently unsearchable and is reported as fine. This contradicts the crash-recovery claim. |
| **F3** | **Silent fake embeddings.** If the sentence-transformers model fails to load, or the OpenAI backend fails, the embedder falls back to SHA-256 hash vectors. The pipeline also pads count mismatches with hash vectors of `""`. | `embedder.py` `embed_chunks` fallback; `researcher_agent.py` alignment block | Hash vectors are 384-dimensional, the same as MiniLM, so no dimension check catches them. The corpus fills with meaningless vectors while ingestion reports success. |
| **F4** | **OpenAI embedding backend is broken.** It is pinned to `openai~=1.87` but calls the pre-1.0 `openai.Embedding.create`, which raises `APIRemovedInV1`. It also makes one API call per text instead of batching. | Verified by calling it | Combined with F3: selecting the OpenAI backend silently produces hash vectors for the whole corpus. |
| **F5** | **No embedding provenance.** Nothing records which model produced a vector. | Schema review | Changing models later mixes incompatible vectors in one store with no way to detect it. |

### 1.4 Hygiene issues (fixed in Phase 0, low effort)

- `requirements.txt` includes `crewai` and `ray`, which are never imported and are very heavy, plus dev tools (`pytest`, `black`, `mypy`, `ruff`) that belong in `requirements-dev.txt`.
- Tool versions are unpinned, so lint counts drift. The CI comment says 40 Ruff / 83 mypy findings; current versions report 76 / 225.
- The `Makefile` `format` and `lint` targets point at a `sera/` directory that does not exist, and call `flake8`, which is not installed.
- There are two arXiv retrievers. `agents/retriever_arxiv.py` is a legacy duplicate with a different `PaperMeta` shape (`id` vs `paper_id`).
- Empty files: the four agent stubs, `eval/metrics.py`, `memory/episodic_db.py`, `docs/design_notes.md`, `scripts/run_local.sh`. **Do not delete the 0-byte `__init__.py` files — they are package markers.**
- The arXiv client has no global rate limiter. arXiv's API terms ask for no more than one request every three seconds. Ingesting ~150 papers needs this; verify the current terms before ingesting.
- `docs/architecture_diagram.md` draws a five-agent system with monitoring and a dashboard that do not exist, with no label saying it is a target.

---

## 2. Resume and README truth-up — do this first

You are interviewing, and the repo is public. This section is the only part of the plan that affects you this week.

### 2.1 Claim check: "Self-Healing Research Automation System"

| Resume claim | Verdict | Action |
|---|---|---|
| "distributed data ingestion and processing pipeline" | **Overclaim.** It is single-process asyncio: concurrent, not distributed. | Say "concurrent". |
| "500+ documents/day", "200 docs/batch, 50ms p95", "no deadlock observed across all test runs" | **Unverifiable.** No benchmark exists. | Remove now. Re-add only the numbers produced by Phase 0.3's benchmark. |
| "global semaphore, shard-level CPU backpressure" | **Half false.** There are three per-stage semaphores. "Shard" and "backpressure" appear nowhere in the code. | Say "per-stage bounded concurrency". |
| "micro-batched embedding pipeline" | **True.** Batches, with retry on `MemoryError`. | Keep. |
| "transactional metadata subsystem using multi-connection SQLite" | **True.** Connection per operation, explicit transactions with rollback, WAL mode. | Keep. |
| "session isolation" | **Overstated.** It is per-operation transactions, not sessions. | Say "per-operation transactions". |
| "heartbeat tracking" | **Does not exist.** Zero matches in the repo. | Remove. |
| "WAL-inspired atomic writes" | **Misdescribed.** It uses SQLite's WAL mode. The real atomic-write work is the downloader's temp-file-and-rename. | Describe the downloader instead. |
| "deterministic recovery under failures (OOM, network, power loss)" | **False today.** F2 proves interrupted papers never recover. | Remove until F2 is fixed and its test is green; then it becomes true. |

> **Correction to v2 of this document:** v2 said the SQLite layer used autocommit "the opposite of session isolation". That was wrong. `isolation_level=None` combined with explicit `BEGIN`/`COMMIT`/`ROLLBACK` is correct manual transaction control. If you edited your resume based on v2, restore the "transactional, multi-connection SQLite" wording.

### 2.2 Bullets that are true today

> **SERA — Research Paper Ingestion & Retrieval Pipeline** (Python, asyncio, SQLite, Chroma)
> - Built an async ingestion pipeline (arXiv → PDF → parse → chunk → embed → vector store) with per-stage bounded concurrency and per-paper failure isolation, so one bad paper never halts a batch.
> - Implemented crash-safe PDF downloads with streaming writes to a temporary file, atomic rename, header validation, and size caps.
> - Designed a typed vector-store abstraction (`Protocol` + error hierarchy) with interchangeable Chroma and JSON backends, enabling hermetic tests with no external services.
> - Built a transactional SQLite metadata layer (WAL mode, per-operation transactions with rollback, foreign-key-enforced paper/chunk/vector linkage).

### 2.3 Bullets that become true after Phase 0.3

> - Made ingestion resumable: papers are marked complete only after their vectors are stored and linked, so an interrupted run is repaired on the next run (covered by a crash-recovery regression test).
> - Eliminated silent embedding fallback and recorded embedding-model provenance per collection, preventing mixed or meaningless vectors.
> - Measured throughput: *[numbers from `docs/benchmarks.md`]*.

Each milestone in §13 adds bullets here. Never add a bullet before the milestone that earns it.

---

## 3. Scope

### 3.1 Scope statement

> Given a research question about ML/AI, SERA plans a multi-step investigation over its paper corpus and live arXiv search, ingests new papers on demand, maintains memory within and across runs, writes a structured report in which every factual claim cites retrieved evidence and every gap is stated, and evaluates that report before returning it.

Check every proposed piece of work against this sentence. If it does not serve the sentence, it is out.

### 3.2 In scope

Planning and replanning · tool use (corpus search, arXiv search, on-demand ingestion, sandboxed code execution) · four-layer memory · in-loop critic · offline evaluation harness · inference cost and latency optimization · tracing and failure taxonomy · fault tolerance with checkpoint/resume · single-node deployment · a measured self-improvement experiment (Milestone 4).

### 3.3 Non-goals

- Training or fine-tuning any model.
- Coupling to your PRM/ORM research (§4, D2).
- A chat UI, multi-user auth, quotas, or billing.
- Multimodal input.
- General-purpose agents; the domain is ML/AI research.
- HuggingFace Hub search. It was in v1/v2; it adds API surface with no eval task that needs it. It is a stretch goal only (§16).
- Multi-node or distributed execution.
- Describing SERA as "self-evolving" before Milestone 4 produces a positive measured result.

### 3.4 Milestones

The plan is cut into four milestones. Each one is independently shippable and interview-usable, so the job search is never waiting on the end of the project.

| Milestone | Phases | What you can truthfully say afterwards | Effort |
|---|---|---|---|
| **M1 — Credible agent** | 0–3 | "Plan/act/critique research agent over a verified corpus, with a calibrated eval harness and a measured baseline." | 51–71 h |
| **M2 — Measured system** | 4–6 | "Four-layer memory and tools, each justified by an eval delta; regression-gated eval." | 26–40 h |
| **M3 — Production-grade** | 7–9 | "Profiled and optimized, chaos-tested, deployed, with a sealed-set final evaluation." | 24–36 h |
| **M4 — Self-improvement** | 10 | "Measured whether reflection-derived lessons improve held-out performance." Either result is publishable. | 8–12 h |
| **Total** | | | **109–159 h** |

This total is higher than v2's 85–115 h. v2 missed the Phase 0 foundation repairs and under-sized the eval set. This estimate is the honest one.

---

## 4. Key decisions

Each decision has a **default** so that work never stalls waiting on it. An ADR is written to confirm or override the default, not to start the question from scratch.

| ID | Decision | Default | Why |
|---|---|---|---|
| **D0** | Package layout | Move everything under a `sera/` package and make it an editable install via `[project]` in `pyproject.toml`. Done once, in Phase 0.2. | Top-level packages named `utils`, `configs`, `memory`, `eval` are collision-prone, imports only work from the repo root, and the Makefile already assumes `sera/`. At 5.8k LOC this is a mechanical, one-commit change; after the agent layer doubles the code it no longer is. |
| **D1** | Orchestration | LangGraph, with nodes written as plain functions over a typed state, so the framework could be swapped out. | Built-in checkpointing and conditional edges; it is already on your resume. Keeping nodes framework-agnostic keeps the interview story about your design, not the library. |
| **D2** | Research coupling | None. The critic uses rules plus an LLM judge. Your PRM is not used. | If the critic underperforms, you must be able to tell whether the architecture or the reward model is at fault. Swapping your PRM in later is a clean ablation. |
| **D3** | Model access | A thin in-repo `LLMGateway` with a role-to-model map (`planner`, `critic`, `synthesizer`, `reflector`, `judge`). One primary provider. A cheap model for development loops, a stronger one for phase-boundary evals. | Changing a model becomes a config change, and cost accounting, retries, and caching live in one place. **The only decision you must make in Phase 0: which provider, and the project's total API budget.** |
| **D4** | Executor | Code, not an LLM. It validates arguments against the tool schema and calls the tool. It makes an LLM call only to repair arguments that fail validation. | An LLM executor doubles calls per step for little gain. The planner already chooses the tool and arguments. |
| **D5** | Critic | Deterministic rules first (empty result, duplicate evidence, schema violation, off-topic by embedding similarity), then the LLM only when the rules are inconclusive. | Most step checks need no LLM. This cuts critic cost several-fold. |
| **D6** | Eval judge vs critic | **Separate components** with separate prompts. Use a different model family for the judge where possible. | If the in-loop critic and the offline judge share a prompt and model, you end up optimizing the agent against its own metric. That is Goodhart's law. |
| **D7** | Episodic store | A separate Chroma collection through the existing adapter, with its own embedding manifest. | Reuses tested infrastructure. A separate collection prevents trajectories from polluting corpus search. |
| **D8** | Sandbox for `code_exec` | Docker: `--network none`, read-only root, non-root user, CPU/memory/pids limits, wall-clock timeout, a fresh container per call. | Strong isolation with no extra service or cost. |
| **D9** | Deployment | One VM (EC2 or GCE) running Docker Compose with a persistent volume. **Not Cloud Run.** | All state lives in SQLite and Chroma files. Cloud Run's filesystem is ephemeral and per-instance, so data would vanish on restart and diverge across instances. Moving to managed Postgres/pgvector is out of scope. |
| **D10** | Self-improvement | Build the Milestone 4 experiment. The name and claim are gated on its result. | It is the most differentiated part of the project, but only if it is measured. |

---

## 5. Architecture

### 5.1 Components

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

`ingest_paper` as a tool is the reason SERA is the right base. When the corpus lacks a paper the question needs, the agent finds it with `arxiv_search`, pulls the full text in through the existing pipeline, and then searches it. That is an honest, concrete capability that most research-agent demos do not have.

### 5.2 Data stores

One store per purpose. Never mix them.

| File | Holds | Written by |
|---|---|---|
| `data/metadata.db` | Papers (with ingestion `status`), chunks (with **full text**), vector links, embedding manifest | Ingestion pipeline |
| `data/chroma/` collection `corpus` | Chunk vectors | Ingestion pipeline |
| `data/chroma/` collection `episodic` | Past-run trajectory summaries | Memory write (Phase 4) |
| `data/memory.db` | Semantic facts and lessons | Memory write, reflector |
| `data/runs.db` | Traces, cost ledger, critic scores | Gateway, orchestrator |
| `data/checkpoints.db` | LangGraph checkpoints | Orchestrator |
| `eval/corpus_manifest.json` (committed) | Exact arXiv IDs + versions + embedding model | Phase 1 |

The corpus snapshot ID is the hash of `corpus_manifest.json`. Every eval result records it.

### 5.3 Control flow

```
INITIALIZE ─ load memory, assemble context
    │
    ▼
PLANNER ─ ordered, tool-typed steps with dependencies
    │
    ▼
┌─▶ budget check (before every node) ──── exceeded ──▶ SYNTHESIZE partial
│       │
│       ▼
│   EXECUTOR ─ validate args → call tool → ToolResult
│       │
│       ▼
│   CRITIC ─ rules → (LLM if inconclusive) → verdict
│       │
│       ├─ accept  ─▶ MEMORY WRITE ─▶ more steps? ─ yes ─┐
│       ├─ retry   ─▶ EXECUTOR (same step, once)          │
│       ├─ replan  ─▶ PLANNER (budgeted)                  │
│       └─ abort   ─▶ SYNTHESIZE partial                  │
└─────────────────────────────────────────────────────────┘
                              │ no more steps
                              ▼
SYNTHESIZER ─ Report with claim-level citations
    │
    ▼
FINAL CHECK ─ every claim cites evidence? gaps stated?
    │          (rules; on failure: one synthesizer revision)
    ▼
RESPONSE
```

### 5.4 Termination conditions

Each one gets its own test with a scripted fake LLM.

| Condition | Default limit | On hit |
|---|---|---|
| Steps | 15 | Synthesize from current evidence, `status: partial` |
| Replans | 3 | Continue with the best plan, confidence capped at `low` |
| Step retries | 1 per step | Mark the step `failed` and replan |
| Wall clock | 240 s | `status: timeout`, return partial |
| Cost | Per-run cap in config (start at $1.00) | `status: budget_exceeded`, return partial |
| On-demand ingests | 3 per run | The tool returns `status: error` with a reason |
| Identical tool call (same tool and normalized args) | 2nd occurrence | Forced replan; a 3rd aborts that branch |
| Consecutive critic rejections | 3 | Abort with diagnosis |

### 5.5 Roles

| Role | Implementation | Does | Must not |
|---|---|---|---|
| Planner | LLM | Decompose the question; assign tools, arguments, and dependencies; replan | Call tools; write prose |
| Executor | Code | Validate, call, normalize, record | Choose tools; judge quality |
| Critic | Rules + LLM | Score each step; give a verdict with a rationale | Rewrite content; call tools |
| Synthesizer | LLM | Write the report from evidence only | Add facts that are not in evidence |
| Reflector | LLM (M4 only) | Turn completed runs into lessons | Run during eval measurement runs |

The roles are disjoint so that every failure can be attributed to exactly one of them.

---

## 6. Contracts

Pydantic models, defined in full in Phase 2 even though some fields stay unused until later. Changing the state schema later is expensive.

### 6.1 AgentState

```python
class AgentState(BaseModel):
    run_id: str
    question: str
    created_at: datetime
    config_snapshot: RunConfig            # models, limits, prompt versions

    plan: list[PlanStep]
    current_step: int
    replan_count: int

    evidence: list[Evidence]              # append-only
    trajectory: list[NodeEvent]           # every node entry/exit
    critic_scores: list[CriticScore]
    memory_context: MemoryContext         # what was injected this step

    budget: BudgetState                   # steps, tokens, cost_usd, elapsed_s, ingests
    status: Literal["running", "complete", "partial", "failed",
                    "timeout", "budget_exceeded"]
    report: Report | None
    failure_reason: str | None
```

Invariants: `evidence` is append-only. The budget is checked before each node, never after. `config_snapshot` makes every run reproducible and attributable.

### 6.2 PlanStep

```python
class PlanStep(BaseModel):
    index: int
    intent: str                           # what this step must establish
    tool: Literal["corpus_search", "arxiv_search", "ingest_paper",
                  "code_exec", "none"]
    args: dict
    depends_on: list[int]                 # enables parallel execution (Phase 7)
    status: Literal["pending", "running", "done", "failed", "skipped"]
```

### 6.3 Evidence

```python
class Evidence(BaseModel):
    id: str                               # "ev-<run>-<n>"
    source_type: Literal["corpus_chunk", "arxiv_abstract", "code_output"]
    source_id: str                        # chunk_id | arxiv id | exec id
    paper_id: str | None
    content: str                          # full chunk text, abstract, or stdout
    retrieved_at: datetime
    step_index: int
    score: float | None                   # retrieval similarity
```

`source_id` for corpus evidence is the existing `chunk_id` in `metadata_db`, so provenance reuses the current schema. This only works once F1 (full chunk text) is fixed.

### 6.4 Tool contract

```python
class ToolSpec(BaseModel):
    name: str
    description: str                      # a prompt; versioned
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    timeout_s: float
    max_retries: int
    fallback: str | None
    side_effects: bool

class ToolResult(BaseModel):
    tool: str
    status: Literal["ok", "empty", "error", "timeout", "rejected"]
    evidence: list[Evidence]
    error: str | None
    latency_ms: int
    attempt: int
    cache_hit: bool
```

**A tool never raises into the orchestrator.** Every exception, including SERA's `VectorStoreError` family, is translated into a `ToolResult` status at the tool boundary. A contract test enforces this for every tool.

### 6.5 Tools

| Tool | Input | Returns | Backed by | Fallback |
|---|---|---|---|---|
| `corpus_search` | `query`, `k`, optional `paper_ids` | Chunk evidence with full text | Vector store + `metadata_db` | none |
| `arxiv_search` | `query`, `max_results` | Abstract evidence | Existing retriever + global rate limiter + response cache | none |
| `ingest_paper` | `arxiv_id` | Count of chunks added; afterwards `corpus_search` can find them | Existing ingestion pipeline | none |
| `code_exec` | `code` (Python) | stdout/stderr evidence | Docker sandbox (D8) | none |

**Tool admission rule:** a tool exists only if at least three dev eval tasks need it. `code_exec` qualifies because §10.1 requires at least three computation tasks. HuggingFace search does not, so it is out.

### 6.6 CriticScore

```python
class CriticScore(BaseModel):
    step_index: int
    decided_by: Literal["rule", "llm"]
    rule_hits: list[str]                  # e.g. ["empty_result", "duplicate_evidence"]
    groundedness: float | None            # 0–1
    relevance: float | None               # 0–1
    novelty: float | None                 # 0–1
    verdict: Literal["accept", "retry", "replan", "abort"]
    rationale: str                        # required, always
```

### 6.7 Report

```python
class Claim(BaseModel):
    text: str
    evidence_ids: list[str]               # must be non-empty for factual claims

class Section(BaseModel):
    heading: str
    claims: list[Claim]

class Report(BaseModel):
    question: str
    summary: str
    sections: list[Section]
    open_questions: list[str]             # what could not be established
    confidence: Literal["high", "medium", "low"]
    limitations: str
```

v2 attached citations to sections. That made citation accuracy unmeasurable, because a section-level citation cannot tell you which claim it supports. Claim-level citations make every claim checkable against the content of the evidence it cites.

---

## 7. Memory

| Layer | Holds | Store | Read | Write |
|---|---|---|---|---|
| Context | Pinned instructions, plan, recent steps, injected memory | Prompt | Every LLM call | Rebuilt every call |
| Episodic | Summaries of past runs: question, plan shape, tools used, outcome | Chroma `episodic` | Top-k at plan time | After each completed run |
| Semantic | Durable facts ("paper X reports Y on Z") and, in M4, lessons | `memory.db` | By key/topic at plan and synthesis time | After an accepted step |
| Corpus (world) | Papers, full-text chunks, vectors | `metadata.db` + Chroma `corpus` | Via tools | Via `ingest_paper` |

### 7.1 Context assembly

| Zone | Starting budget | Contents | Overflow rule |
|---|---|---|---|
| Pinned | ~25% | System prompt, tool specs, question, current plan | Never evicted. If pinned content alone overflows, fail the configuration check at startup. |
| Retrieved memory | ~20% | Episodic hits, semantic facts | Drop the lowest-scored items |
| Recent steps | ~40% | Last K steps with their evidence | Evict the oldest into Compressed |
| Compressed | ~15% | Summary of evicted steps | Re-summarize |

The percentages are starting points, to be tuned against the eval. Budgets are enforced in code. **Every eviction is logged as an event.** Silent truncation is the most common cause of unexplained quality drops.

### 7.2 Memory rules

1. **Each layer must justify itself** with an eval delta above the noise floor (§10.6). A layer that does not help is reported as such, not quietly kept.
2. **Eval isolation.** Every eval task starts from the same frozen memory snapshot, and writes made during an eval run are discarded. Without this, a later task benefits from an earlier task's memory, and re-running the suite retrieves the previous run's trajectory for the same question. The scores would inflate and mean nothing.

---

## 8. LLM gateway and prompts

- **One interface:** `complete(role, messages, output_model) -> (parsed, LLMCall)`. Each role maps to a model in config.
- **Structured output:** the provider's native schema or tool-calling mode, then Pydantic validation, with one repair attempt that feeds the validation error back. Two failures produce a typed error, not an exception.
- **Retries:** exponential backoff with jitter on rate-limit and 5xx errors; no retries on 4xx validation errors.
- **Timeouts:** enforced per call and separately from the run-level clock.
- **Cost ledger:** every call writes tokens, model, latency, cost, role, and prompt version to `runs.db`.
- **Prompt caching:** prompts are built with the stable prefix first (system prompt, then tool specs, then pinned context), so the provider's prompt caching applies. Exactly how caching is enabled depends on the provider; confirm it in the provider docs when making D3.
- **Prompts as files:** `sera/prompts/<role>.md`, with a version header. The version is logged on every call. A prompt change is treated like a code change: it needs an eval run before and after.
- **Development mode:** the cheap model plus a response cache keyed by (model, prompt version, messages hash), so iterating on downstream code does not re-spend tokens on upstream calls.

---

## 9. Observability, reliability, safety

### 9.1 What is recorded

`runs`: id, question, status, cost, latency, config snapshot, corpus snapshot ID
`node_events`: run, node, start, end, input/output hashes, error
`llm_calls`: run, role, model, prompt version, tokens, cost, latency, cache hit
`tool_calls`: run, tool, args hash, status, latency, attempt, cache hit
`critic_scores`: as in §6.6

The dashboard is a static HTML page generated from `runs.db`: eval scores by phase, cost and latency distributions, failure taxonomy, and critic verdict mix.

### 9.2 Failure taxonomy

Every failed or partial run is assigned exactly one primary code.

| Code | Meaning | Usual fix |
|---|---|---|
| `F-PLAN` | Plan wrong or malformed for the question | Planner prompt and examples |
| `F-TOOL-SEL` | Wrong tool for a correct step | Tool descriptions |
| `F-TOOL-ARG` | Invalid arguments after repair | Schema, examples |
| `F-TOOL-EXT` | External failure (arXiv, Docker, provider) | Retry, cache, fallback |
| `F-RETRIEVAL` | Right query, wrong or insufficient chunks | Chunking, k, query rewrite |
| `F-CTX` | Overflow or eviction lost needed content | Assembly budgets |
| `F-GROUND` | Claim not supported by its cited evidence | Synthesizer constraints, final check |
| `F-LOOP` | Repetition | Loop detection |
| `F-BUDGET` | Ran out of steps, time, or cost | Planning efficiency |
| `F-JUDGE` | The eval judge was wrong | Rubric, recalibration |

`F-RETRIEVAL` is new in v3. In a corpus-backed agent it is usually the largest category, and it was missing.

### 9.3 Reliability patterns

| Pattern | Applies to | Status |
|---|---|---|
| Backoff with jitter | LLM, arXiv, PDF fetch | Partly exists (retriever, downloader) |
| Global rate limiter | arXiv | New (Phase 0.3) |
| Circuit breaker | Each external dependency | New (Phase 8) |
| Nested timeouts | Call, tool, node, run | New |
| Checkpoint per node, resume endpoint | Orchestrator | New (Phase 3) |
| Idempotent side effects | `ingest_paper`, memory writes | Repaired in Phase 0.3 |
| Atomic file writes | Downloads, reports | Exists (downloader pattern) |
| Graceful partial results | Every terminal path | New |

### 9.4 Chaos suite (Phase 8)

Each scenario is an automated test asserting a categorized, graceful outcome:

1. A tool returns an error mid-run
2. A tool times out
3. A tool returns nothing
4. The LLM returns invalid structured output twice
5. The LLM requests a nonexistent tool
6. Context is forced to overflow
7. The process is killed mid-run → resume completes from the last checkpoint
8. The cost cap is hit mid-run
9. A retrieved chunk contains a prompt-injection payload
10. Concurrent runs write the same semantic-memory key
11. Ingestion is killed mid-paper → the next run repairs it (the F2 regression test, extended)

Scenarios 1–6, 8, and 10 run in normal CI with fakes. 7, 9, and 11 run in the integration tier.

### 9.5 Prompt injection

- Retrieved text is placed in delimited data blocks, never in the instruction region.
- Tool choice comes from the plan, never from retrieved text.
- `code_exec` code is written by the planner from the question and step intent. Retrieved text is passed to it only as data input, never executed.
- An injection payload is included in the dev adversarial tasks, and a different one in the sealed set.
- Steps where retrieved content contains imperative patterns aimed at the agent are logged.

### 9.6 Secrets and spend

`.env` only, with `.env.example` committed. The run-level cost cap is enforced in the gateway. **A project-level budget is enforced in the gateway too**: it refuses calls once the total in `runs.db` passes the budget you set in D3. A runaway loop is stopped by your code, not by your credit card.

---

## 10. Evaluation

### 10.1 Task sets

| Set | Size | Purpose | Who sees it |
|---|---|---|---|
| **Dev** | 30 tasks | Development, ablations, regression | Used constantly |
| **Smoke** | 8 tasks, a fixed subset of Dev | Fast check after a change | Used constantly |
| **Sealed** | 8 tasks (2 per tier) | One-time generalization check | Opened once, at the end of Phase 9 |
| **Experience** | 15 tasks, written in Phase 10 | Lesson generation for M4 only | Never scored |

Dev composition: Tier 1 (single-hop factual) 8 · Tier 2 (multi-hop synthesis) 10 · Tier 3 (open-ended, evidence limits) 8 · Tier 4 (adversarial: false premise, nonexistent paper, injection) 4. Across Tiers 2 and 3, at least 3 tasks require computation (e.g. relative improvement from reported numbers), and at least 3 require a paper that is **not** in the corpus, which exercises `ingest_paper`.

**Why 30, not v2's 12:** with 12 tasks, one task flipping moves completion by 8 points, so the per-layer "eval delta" rule could not distinguish an improvement from noise. 30 tasks, each run 3 times, gives the rule real meaning at a cost you can afford with the dev model (§10.7).

**Why adversarial tasks are in Dev too:** in v2 they existed only in the sealed set, so refusal behavior was measured exactly once, never developed, and most likely failed with no recourse. Now it is developed on Dev and validated on fresh sealed cases.

**Order matters:** build the corpus first, then write tasks against it. A task whose `must_cite` paper is not in the corpus (and is not meant to require `ingest_paper`) measures ingestion coverage, not reasoning.

### 10.2 Sealed set handling

Write it in Phase 1. Store it outside the repo, or encrypted in it. Commit its SHA-256 hash in Phase 1 so that you can later prove it was not changed after you saw results. Open it once, at the end of Phase 9. After that it is no longer sealed; any later work reports against Dev only.

**Single-author bias.** You write the tasks, label the judge calibration set, and build the agent, so your blind spots enter all three. Writing the tasks before the planner exists limits this. If a peer can write even 4 of the 8 sealed tasks without seeing the agent, the sealed result becomes much stronger evidence. Record in the results file who wrote each sealed task.

### 10.3 Task schema

```json
{
  "id": "t2-004",
  "tier": 2,
  "question": "...",
  "requires": ["corpus_search", "ingest_paper"],
  "must_cite": ["2305.20050"],
  "must_not_claim": ["that process supervision was evaluated on code generation in this paper"],
  "rubric": {"completeness": "states both the supervision signal and the dataset"},
  "expect_uncertainty": false,
  "step_range": [2, 6],
  "notes": "catches cross-paper attribution errors"
}
```

Write `must_not_claim` from the hallucination you expect, not from the answer you want.

### 10.4 What "pass" means for a task

A run passes only if **all** of these hold:

1. `status == complete` (or, for `expect_uncertainty` tasks, `complete` or `partial` with the gap stated in `open_questions`)
2. Every `must_cite` paper appears among the cited evidence
3. No `must_not_claim` is asserted (checked by the judge)
4. Citation accuracy for the run is at least 0.9 (§10.5)
5. Rubric score from the judge is at least 4/5
6. Tier 4: the false premise is rejected or the injection is ignored, and nothing is fabricated

A task's pass rate is its passes divided by k runs.

### 10.5 Metrics

| Level | Metric | Computed by |
|---|---|---|
| Output | **Pass rate** (headline number) | §10.4 |
| Output | Citation accuracy: the share of claims whose cited evidence entails the claim | Judge, per claim, against full evidence content |
| Output | Citation coverage: the share of factual claims with at least one citation | Code |
| Output | Unsupported claims per report | Judge |
| Output | Uncertainty correctness on `expect_uncertainty` tasks | Judge + code |
| Trajectory | Steps, replans, redundant calls, ingests used | Code from `runs.db` |
| Trajectory | Memory hit rate: injected memory items the planner actually used | Code + judge |
| Turn | Tool-selection agreement with `requires` | Code |
| Turn | Argument validity rate, critic verdict mix | Code |
| Cost | Cost and latency per task (p50/p95) | `runs.db` |

### 10.6 Noise floor and the decision rule

In Phase 2, run the baseline on Dev **three times**. Record the per-task pass-rate spread and the standard deviation of each continuous metric. That is the noise floor.

**A change counts as an improvement only if both hold:**
- The headline pass rate improves by more than 2× its baseline run-to-run standard deviation, **or** at least 3 more tasks reach pass rate ≥ 2/3, and
- No Tier 1 task regresses.

Anything else is reported as "no measurable effect". Writing that down is a valid, useful result.

### 10.7 Judge

- A separate prompt and, where possible, a different model family from the generator (D6). The judge model is pinned. Changing it starts a new results series.
- Position bias: randomize order in any pairwise comparison. Verbosity bias: the rubric explicitly ignores length.
- **Calibration** (Phase 2, after the skeleton produces real outputs): hand-label 40 claim-level citation judgements and 20 task-level pass/fail outcomes. Measure agreement between you and the judge. **If agreement on either is below 80%, fix the rubric before trusting any judge-derived number.** Re-calibrate in Phase 6 and whenever the judge prompt changes.
- v2 scheduled calibration in Phase 1, before any agent existed to produce outputs to label. That was an ordering error.

### 10.8 Reproducibility

- Every result records: git SHA, corpus snapshot ID, prompt versions, model IDs, judge version, k, date.
- Eval runs `arxiv_search` against a **recorded cassette** of responses captured the first time each query runs, so live arXiv changes cannot move the scores. `ingest_paper` during eval ingests into a throwaway copy of the corpus, which is discarded afterwards.
- LLM outputs are not cached during eval; their variance is what k = 3 measures.

### 10.9 Eval cost

```
full Dev eval ≈ 30 tasks × 3 runs × (agent cost per task + judge cost per task)
smoke eval    ≈ 8 tasks × 1 run × (same)
```

Put your real per-task cost from Phase 2 into this formula and fix a schedule that fits your D3 budget. The intended schedule: **full eval at each phase exit and for each ablation, smoke eval after meaningful changes.** Development iteration uses the cheap model. The stronger model is used only for phase-exit runs, and all runs within one comparison must use the same model.

### 10.10 Where eval runs

| Tier | Trigger | Runs | Needs a real LLM |
|---|---|---|---|
| CI (every push/PR) | Automatic | Format, lint, type-check of new code, unit + contract + graph tests with a scripted `FakeLLM`, most chaos scenarios | No |
| Integration | Manual `workflow_dispatch` | Smoke eval with the dev model, cost-capped | Yes (repo secret) |
| Phase exit | You, locally | Full Dev eval, k = 3; results committed | Yes |

v2's "CI runs the eval suite on every push" was not workable. It would have meant real API spend and flaky results on every commit, with secrets exposed to every branch. The regression gate is the integration tier, run before merging work that touches prompts, tools, or the graph.

---

## 11. Testing strategy (agent layer)

| Kind | What | Tool |
|---|---|---|
| Unit | Context assembly budgets, rule-based critic, budget accounting, loop detection | pytest |
| Contract | Every tool returns a `ToolResult` and never raises, including on injected exceptions | pytest, parametrized over tools |
| Graph | Each termination condition, retry, replan, and abort path, driven by a `FakeLLM` that replays scripted outputs | pytest + `FakeLLM` |
| Golden trace | Two or three recorded full runs replayed deterministically to catch unintended graph-behavior changes | pytest + fake gateway |
| Chaos | §9.4 | pytest |
| Ingestion | Existing suite + `test_ingest_crash_recovery.py` + a no-silent-shim test | pytest |
| Eval | §10 | harness |

`FakeLLM` is built in Phase 2, before any real node is written. It is what makes the agent layer testable in CI without spending money.

Lint policy: Ruff and mypy-strict run in CI on new packages (`sera/agents`, `orchestrator`, `tools`, `gateway`, `memory/episodic.py`, `memory/semantic.py`, `memory/context.py`, `evaluation`, `observability`) from their first commit. Legacy modules are brought in one at a time as they are touched. Tool versions are pinned in `requirements-dev.txt`.

---

## 12. Target repository layout

```
SERA/
├── pyproject.toml             # [project] + tool config; `pip install -e .`
├── requirements.txt           # runtime only
├── requirements-dev.txt       # pinned test/lint tooling
├── Makefile                   # working targets
├── docker-compose.yml         # Phase 9
├── .env.example
├── README.md                  # what works today; status table
├── docs/
│   ├── design_doc.md          # this file
│   ├── architecture.md        # diagram with built/planned status (replaces architecture_diagram.md + _partB.md)
│   ├── benchmarks.md          # ingestion throughput (Phase 0.3)
│   ├── eval_report.md         # results across phases
│   └── failure_analysis.md
├── adr/                       # ADR-000 … ADR-010, one per decision in §4
├── eval/                      # DATA ONLY (no code)
│   ├── corpus_manifest.json
│   ├── tasks/dev_v1.json
│   ├── tasks/sealed_v1.sha256 # hash only; sealed tasks kept outside repo
│   ├── cassettes/
│   └── results/<date>_<phase>_<sha>.json   # never deleted
├── sera/
│   ├── configs/
│   ├── ingestion/
│   │   ├── pipeline.py        # was agents/researcher_agent.py (IngestionPipeline)
│   │   └── arxiv_retriever.py # was agents/retriever_arxiv_real.py
│   ├── parsers/  utils/  embeddings/  services/   # moved as-is
│   ├── memory/                # existing + episodic.py, semantic.py, context.py
│   ├── tools/                 # contract.py, registry.py, one module per tool
│   ├── gateway/               # llm.py, cost.py, cache.py
│   ├── prompts/               # <role>.md, versioned
│   ├── agents/                # planner.py, executor.py, critic.py, synthesizer.py, reflector.py
│   ├── orchestrator/          # state.py, graph.py
│   ├── evaluation/            # harness, judge, metrics
│   ├── observability/         # trace store, dashboard generator
│   └── api/                   # FastAPI app
├── scripts/                   # thin CLIs over sera.*
└── tests/
    ├── unit/  contract/  graph/  chaos/  ingestion/  integration/
    └── fakes/                 # FakeLLM, fake tools
```

Rule: `eval/` holds data and `sera/evaluation/` holds code. The repo will never again have a package and a data directory with the same name.

---

## 13. Phase plan

Each phase lists its work, exit criteria, and effort. **Exit criteria are pass/fail; "mostly done" is not done.**

### Milestone 1 — Credible agent

#### Phase 0.1 — Truth-up (2–3 h) · do this week

- Resume: apply §2.1; use the §2.2 bullets.
- README: add a status table (built / in progress / planned). Remove nothing that is true.
- Merge `architecture_diagram.md` and `architecture_partB.md` into `docs/architecture.md` with the §5.1 diagram and its `[B]/[R]/[N]` legend.
- Commit this file as `docs/design_doc.md`; delete the empty `docs/design_notes.md`.

*Exit:* every statement in the README, the docs, and the resume is true at the current commit.

#### Phase 0.2 — Restructure (3–4 h)

- Record a baseline: `pytest` and the hermetic run both green (they are, as of `3149561`).
- One commit: create the `sera/` package (D0), move modules per §12, rename `ResearcherAgent` → `IngestionPipeline`, delete the legacy `retriever_arxiv.py`, delete the non-`__init__` empty files (§1.4), add `[project]` to `pyproject.toml`, fix the Makefile, split and pin requirements (drop `crewai` and `ray`).
- Update CI paths. Run the full suite. If anything is red, fix it or revert; do not push a red commit.
- Write ADR-000.

*Exit:* `pip install -e .` works from a clean venv; the same tests pass as before; CI is green; `make lint` and `make format` run.

#### Phase 0.3 — Foundation repair (10–14 h)

| Fix | Work | Done when |
|---|---|---|
| F1 | Add a `text` column to `chunks` (schema version table + migration); store full chunk text; add the text to vector metadata or fetch it by `chunk_id` | A stored chunk's full text round-trips |
| F2 | Add a `status` column (`pending` / `complete`) to `papers`; set `complete` only after vectors are stored, linked, and persisted; skip only `complete` papers; on retry, remove stale chunks and vectors first | `test_ingest_crash_recovery.py` is green |
| F3 | The shim backend is allowed only when explicitly configured (tests); otherwise a load or embed failure raises and fails the paper; remove padding with hash vectors | A test proves that a model load failure fails ingestion loudly |
| F4 | Port to the `openai>=1` client with real batched requests, or remove the backend. Keep it only if you will use it. | Backend works or is gone |
| F5 | An embedding manifest per collection (model ID, dimension, normalization); adapters refuse vectors from a different manifest | Mixed-model insert is rejected by a test |
| — | Global arXiv rate limiter (one request per 3 s; verify current terms) | Unit test on limiter timing |
| — | `scripts/benchmark_ingest.py`: ingest N papers with the real local model; report papers/min and p50/p95 per stage, with hardware noted | `docs/benchmarks.md` committed |

Then update the resume with the §2.3 bullets, filled in with the real benchmark numbers.

*Exit:* all fix criteria met; benchmark committed; resume updated from the benchmark only.

#### Phase 1 — Corpus and eval set (14–20 h)

Writing 38 tasks with `must_not_claim` lists means reading the source passages behind each one. Budget ~15 minutes per task; this is the most underestimated part of the plan.

- Choose ~150 papers (your nine-track reading list). Write `eval/corpus_manifest.json` with versioned arXiv IDs. Ingest it. Record the snapshot ID.
- Write the 30 Dev tasks against the corpus (§10.1, §10.3). Define the 8-task Smoke subset.
- Write the 8 Sealed tasks, store them outside the repo, and commit their hash.
- Make decision D3: provider, role→model map, project budget. Write ADR-001, 002, 003, and 008.

*Exit:* corpus ingested with no failed papers; task files validate against the schema; sealed hash committed; budget set.

#### Phase 2 — Gateway, query path, skeleton, calibration (10–14 h)

- `LLMGateway` with structured output, retries, cost ledger, dev response cache, and project-budget guard. Build `FakeLLM` at the same time.
- All §6 contracts in full. `corpus_search` tool + contract test.
- Skeleton graph: question → `corpus_search` → synthesizer → report. No planner yet.
- API: `POST /research`, `GET /runs/{id}`.
- Eval harness v0: runs Dev k = 3, computes the §10.5 metrics, writes a results file.
- **Run the baseline three times → record the noise floor (§10.6).**
- **Judge calibration (§10.7)** on the skeleton's outputs.

*Exit:* baseline and noise floor recorded; judge agreement ≥ 80% on both label sets; CI green with `FakeLLM` graph tests.

#### Phase 3 — Planner, executor, critic, checkpointing (12–16 h)

- Planner with a typed plan; code executor (D4); critic rules plus LLM (D5); replan and retry edges.
- All §5.4 termination conditions, each with a `FakeLLM` graph test.
- LangGraph checkpointing and `POST /runs/{id}/resume`.
- `arxiv_search` tool with cassette support.
- ADR-007 (executor as code).
- Full Dev eval, compared with the Phase 2 baseline under the §10.6 rule.

*Exit:* every termination test green; resume-after-kill works; the eval delta is recorded, whatever it is.

**Retrieval decision point.** The embedding model (MiniLM-L6) is small and not trained on scientific text. If `F-RETRIEVAL` accounts for more than 40% of failed runs at this exit, retrieval is the bottleneck and memory work in Phase 4 will not show real gains. In that case, write an ADR and pull the cross-encoder reranker (§16) forward into Phase 4, before 4a.

**→ Milestone 1 reached.** Update the resume and README. This is a complete, honest, interview-ready agentic system.

### Milestone 2 — Measured system

#### Phase 4 — Memory (12–16 h)

Add one layer at a time. Run a full Dev eval after each one and apply the §10.6 rule. Eval isolation (§7.2) must be in place first.

- 4a Context assembly with zone budgets and eviction events
- 4b Episodic memory
- 4c Semantic facts
- ADR-006 (episodic store)

*Exit:* three recorded deltas. Layers that did not help are documented, and either kept off by default or removed.

#### Phase 5 — On-demand ingestion and code execution (6–12 h)

- `ingest_paper` tool over `IngestionPipeline`, capped per run, using the throwaway corpus copy during eval.
- `code_exec` in the Docker sandbox (D8), with sandbox-escape tests (network attempt, filesystem write, fork bomb, timeout).
- ADR-004 (sandbox).

*Exit:* contract tests green; sandbox tests green; tasks that require these tools show a measured change.

#### Phase 6 — Eval hardening (8–12 h)

- Harness: three metric levels, historical comparison table, per-task flip report between any two results files.
- Integration CI tier with the smoke eval (§10.10).
- Judge re-calibration.
- `docs/eval_report.md` covering Phases 2–6.

*Exit:* `make eval` and `make eval-smoke` work; the comparison report is generated automatically.

**→ Milestone 2 reached.**

### Milestone 3 — Production-grade

#### Phase 7 — Inference optimization (8–12 h)

Profile first, using `runs.db`, to find where tokens, money, and time actually go. Then, in order of measured leverage:

- Prefix-ordered prompts with provider caching
- Parallel execution of independent steps (`depends_on`)
- Tool-response caching for deterministic calls
- Critic rule coverage, to reduce LLM critic calls
- Streaming events over SSE

*Exit:* a before/after table for cost and p50/p95 latency, **with the Dev pass rate not lower beyond the noise floor**. A faster but worse system is a regression.

#### Phase 8 — Fault tolerance (8–12 h)

- Circuit breakers and nested timeouts.
- The full §9.4 chaos suite; every scenario gets a failure code from the §9.2 taxonomy.
- ADR on the failure-handling policy.

*Exit:* all 11 chaos scenarios pass.

#### Phase 9 — Deployment and final evaluation (8–12 h)

- Dockerfile and `docker-compose.yml` with a persistent volume; deploy to one VM (D9); ADR-005.
- The static dashboard generated from `runs.db`.
- **Open the sealed set. Run it once, k = 3. Record the result whatever it is.**
- Final `docs/eval_report.md` and `docs/failure_analysis.md`.

*Exit:* live endpoint; sealed result recorded; docs complete; resume and README updated from recorded numbers only.

**→ Milestone 3 reached.**

### Milestone 4 — Self-improvement experiment

#### Phase 10 — Reflection, measured (8–12 h)

- Write the 15 Experience tasks (disjoint from Dev and Sealed).
- Reflector: after each Experience run, write structured lessons to semantic memory (`when <question pattern>, <tactic>, because <evidence from trajectory>`).
- The planner retrieves applicable lessons at plan time.
- **Experiment:** Dev with the lesson store empty (cold) vs populated from Experience runs only (warm), k = 3 each, decided by the §10.6 rule. Never generate lessons from Dev runs; that measures memorization, not improvement.
- ADR-010.

*Exit:* the result is recorded and written up. **Only if warm beats cold above the noise floor may SERA be called self-improving**, and then with the number attached.

---

## 14. ADR log

Template (half a page each):

```
# ADR-NNN: <title>
Date · Status: proposed | accepted | superseded by ADR-MMM
Context — what forces the decision; constraints
Options — each with pros and cons
Decision — what was chosen (default from §4 or override)
Consequences — what gets easier, what gets harder, what would trigger revisiting
```

| ADR | Topic | Default (§4) | Written in |
|---|---|---|---|
| 000 | Package layout | D0 | Phase 0.2 |
| 001 | Orchestration | D1 | Phase 1 |
| 002 | Critic design and judge separation | D5, D6 | Phase 1 |
| 003 | Model gateway and provider | D3 | Phase 1 |
| 004 | Code sandbox | D8 | Phase 5 |
| 005 | Deployment topology | D9 | Phase 9 |
| 006 | Episodic store | D7 | Phase 4 |
| 007 | Executor as code | D4 | Phase 3 |
| 008 | Research decoupling | D2 | Phase 1 |
| 009 | Failure-handling policy | — | Phase 8 |
| 010 | Self-improvement result and naming | D10 | Phase 10 |

---

## 15. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Unverifiable claims found in an interview | High until 0.1 | Severe | Phase 0.1 this week; rule 1 of §0 |
| Agent built on a corrupt corpus (F1–F5) | Certain if skipped | Severe | Phase 0.3 before any agent code |
| Eval deltas indistinguishable from noise | High | High | 30 tasks, k = 3, noise floor, decision rule |
| Memory leakage inflating scores | High | High | Per-task memory snapshot; Experience split for M4 |
| Judge unreliable | Medium | High | Calibration with an 80% gate; pinned judge |
| Critic and judge collusion (Goodhart) | Medium | High | Separate components (D6) |
| API spend overrun | Medium | Medium | Dev model, response cache, run and project caps in the gateway |
| Restructure breaks things | Low | Medium | Tests green before and after; one commit; revert if red |
| arXiv throttling or blocking during ingest | Medium | Medium | Global rate limiter, cassettes, response cache |
| Timeline pressure from the job search | High | Medium | Milestones are independently shippable; M1 alone is interview-complete |
| Framework API churn | Medium | Low | Pinned versions; framework-agnostic nodes |
| Deploying file-based state to an ephemeral platform | Medium | High | D9: single VM with a persistent volume |

---

## 16. Stretch goals

Only after M3, and only if a Dev task needs them: HuggingFace Hub search · a cross-encoder reranker for `corpus_search` · a better scientific-text embedding model (requires re-embedding under a new manifest) · self-hosted inference with SGLang/vLLM for prefix-cache experiments · a PRM-based critic ablation using your own research model (D2's follow-up experiment).

---

## 17. Prompt templates for unblocking

Paste the named sections plus your specifics.

**Design question**
> Context: SERA design doc §5 and §6 [paste]. I am implementing <component> in Phase <N>, whose exit criteria are <paste>. Options: A <…>, B <…>. Which better meets the exit criteria, and what failure mode am I not seeing? Do not suggest adding components the doc lists as non-goals (§3.3).

**Debugging**
> Context: §6 contracts and the §9.2 taxonomy [paste]. On task <id> (tier <t>) the run did <behavior>. Trajectory and `runs.db` excerpt: [paste]. Classify it with one failure code, and give the smallest fix that adds no new architecture.

**Eval task review**
> Context: §10.3 and §10.4 [paste]. Draft task: [paste JSON]. Would `must_not_claim` actually fire on the hallucination this task is meant to provoke? Is the task answerable from the corpus manifest [paste IDs]?

**Optimization**
> Context: §5.1, §8, and this profile from `runs.db` [paste]. I want lower p95 latency and cost with the Dev pass rate unchanged beyond the noise floor of <x>. What has the highest leverage, and how do I verify there was no quality regression?

**Adversarial review — use at every phase exit**
> Act as a principal engineer reviewing Phase <N> of this design [paste the phase + relevant sections + the results file]. Do not agree with me. Find the assumption most likely to be wrong, any way the eval could be inflated, and the failure mode I have not tested.

---

## 18. Interview map

| Question | Where the answer lives |
|---|---|
| "Walk me through your agent's architecture." | §5, ADRs |
| "How do you manage context?" | §7.1 |
| "How do you know it works?" | §10, especially the noise floor and judge calibration |
| "How do you stop it looping or overspending?" | §5.4, §9.6 |
| "How do tools fail?" | §6.4, §9.3, §9.4 |
| "How did you make it cheaper or faster?" | Phase 7 before/after table |
| "Does the memory actually help?" | Phase 4 deltas, including the negatives |
| "Is it really self-improving?" | Phase 10 result |
| "What did you get wrong?" | §1.3 (F1–F5 found in your own code), §2.1, the v2 correction notes |

The last row is a strength, not a weakness. Auditing your own system, finding silent data corruption, and proving it with a test is the strongest staff-level story in this project.

---

## 19. Start here

1. Phase 0.1 — resume, README status table, architecture labelling, commit this file. *(This week.)*
2. Phase 0.2 — restructure in one green commit.
3. Phase 0.3 — add `test_ingest_crash_recovery.py` (red), then fix F1–F5 until it and the new tests are green; then run the benchmark.
4. Phase 1 — make decision D3 (provider and budget), then build the corpus and tasks.

No agent code is written before step 3 is finished.

---

## Appendix A — Changes from v2 and why

| v2 said | v3 says | Reason |
|---|---|---|
| SQLite layer is autocommit, "opposite of session isolation" | It is correct per-operation transactional design | v2 misread `isolation_level=None`; every write is wrapped in explicit `BEGIN`/`COMMIT`/`ROLLBACK` |
| `corpus_search` is a thin wrapper | It requires the F1 schema fix first | Full chunk text is never stored |
| "Delete the eight 0-byte files" | Delete the 8 listed non-`__init__` empty files; keep `__init__.py` | 18 files are 0 bytes; 10 of them are package markers |
| 17 test modules | 18 modules, 11 run in CI (37 tests) | Recounted |
| 12 eval tasks; adversarial only in the sealed set | 30 Dev + 8 Sealed; adversarial tasks in both | 12 tasks cannot separate effects from noise; refusal behavior must be developed, not just tested once |
| Judge calibration in Phase 1 | Phase 2, on real outputs | Nothing exists to label in Phase 1 |
| No rule for what an "eval delta" means | Noise floor + explicit decision rule (§10.6) | Otherwise any delta can be claimed as an improvement |
| Episodic memory and reflection measured on the same tasks they learn from | Per-task memory snapshots; a separate Experience split | v2's design would have measured memorization and called it improvement |
| CI runs the eval on every push | Three tiers; real-LLM eval is manual and cost-capped | Cost, flakiness, secrets |
| Section-level citations | Claim-level citations | Citation accuracy must be computable per claim |
| LLM executor | Executor is code | Halves calls per step for little loss |
| Critic is an LLM judge | Rules first, then LLM; separate from the eval judge | Cost; Goodhart |
| Tools include `hf_search` and `report_writer` | Out; `ingest_paper` added | No task needs HF; the synthesizer already produces the report as structured output; on-demand ingestion reuses SERA's strongest component |
| Deploy to Cloud Run or EC2 | A single VM with a persistent volume | All state is file-based; Cloud Run's filesystem is ephemeral |
| Move parsers/utils/services into `ingestion/` | Move everything under a `sera/` package, unchanged internally | Cleaner namespace, Makefile already expects it, done once while the codebase is small |
| 85–115 h | 109–159 h, split into four milestones | v2 missed the foundation repairs and under-sized the eval |
| Embedding pipeline described as good | Silent hash-vector fallback (F3), broken OpenAI backend (F4) | Found by reading the fallback path and calling the API |
