```mermaid
flowchart TD
    A[User Query / API Request] --> B[FastAPI Interface]
    B --> C[LangGraph Orchestrator]

    subgraph AGENTS[Agent Layer]
        R1[Researcher Agent]
        R2[Reader Agent]
        R3[Critic Agent]
        R4[Synthesizer Agent]
        R5[Reflection Agent]
    end

    C --> R1
    C --> R2
    C --> R3
    C --> R4
    C --> R5

    R1 --> M[Memory System]
    R2 --> M
    R3 --> M
    R4 --> M
    R5 --> M

    subgraph MEMORY[Knowledge Memory Layer]
        M1[Vector Store (Chroma/Weaviate)]
        M2[Episodic DB (SQLite)]
    end

    M --> O[Observation / Monitoring (W&B, Prometheus)]
    O --> D[Streamlit Dashboard / UI]
```
