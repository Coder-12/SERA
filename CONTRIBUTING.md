# 📄 **CONTRIBUTING.md (Production-Grade)**

```markdown
# Contributing to SERA (Scientific Exploration & Research Assistant)

Thank you for your interest in contributing to **SERA** — an advanced multi-agent
AI system for end-to-end scientific research automation.

This document explains the guidelines, workflows, and standards that ensure the
project maintains **high quality**, **reproducibility**, **clarity**, and **professionalism**.

---

# 🧭 Project Philosophy

SERA is designed with:

- **Scientific rigor**
- **Deterministic reproducibility**
- **Transparent model behavior**
- **High-quality engineering**
- **Production-grade architecture**
- **Benchmarkability**

We encourage contributions in a way that sustains these values.

---

# 📁 Repository Structure

sera/
agents/ # Research agent, future multi-agent components
memory/ # Vector stores, Chroma adapter, simple adapter
embeddings/ # Embedding backends
utils/ # Chunker & utilities
parsers/ # PDF parsing logic
services/ # PDF Downloader
tests/ # Unit, integration & live tests
configs/
scripts/
README.md
CHANGELOG.md
CONTRIBUTING.md
```


Each directory corresponds to a module with clear responsibilities.
Please keep new files consistent with the structure and naming conventions.

---

# 🧪 Testing Requirements

Every PR **must pass all tests**:

### ✔ Unit Tests

```
pytest -q

```

### ✔ Integration Tests

```
pytest tests/*integration*

```

### ✔ Live Tests (optional, slow)

```
pytest tests/test_full_integration_live.py -s

```

Live tests hit arXiv and real ChromaDB, so they may be skipped unless needed.

### ⚠ Required
- Any new feature **must include new tests**.
- No PR will be accepted without **complete test coverage** for new code.

---

# 🧹 Code Style Guidelines

We strictly follow:

- **PEP8**
- **Black** (code formatting)
- **isort** (import sorting)
- **docstring standards** (NumPy / Google style)
- **typing / mypy where reasonable**
- **logging over print**
- **no silent error suppression without logs**

### Format Code

```
black .
isort .

```

---

# 🔧 Environment Setup

### Clone repository

```
git clone [https://github.com/](https://github.com/)<yourname>/sera.git
cd sera

```

### Create environment

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

### Install dev requirements

```
pip install -r requirements-dev.txt

```

---

# 🔀 Branching Strategy

We use a professional Git workflow:

```
main # stable, production-ready
dev # staging (all development happens here)

dev/v0 # version 0 cluster (Research Agent)
dev/v0.1 # patch releases
dev/v0.2 # next feature series
dev/v1 # multi-agent architecture (future)

```

## ✔ Creating a new branch

```
git checkout dev
git pull
git checkout -b dev/<version>/<feature-name>

```

---

# 📝 Commit Message Convention

We follow **Conventional Commits**:

```
feat: add new chunker boosting logic
fix: correct chroma adapter DIM mismatch
docs: update README with examples
test: add new test case for PDF parser
chore: bump versions and update deps
refactor: redesign embedding batching logic

```

Examples:

```
feat: add section-aware chunking support
fix: robustify PDFParser for irregular PDFs
test: add tests for retry logic in embedder

```

---

# 🔁 Pull Request Workflow

1. **Create a feature branch**
2. Run full test suite
3. Update documentation when necessary (README, CHANGELOG)
4. Submit PR into `dev`
5. Address review feedback
6. PR gets merged into `dev`
7. `dev` → `main` when version release is prepared

PR requirements:

- Tests must pass
- Code must be formatted
- No commented debugging blocks
- No dead code
- Descriptive PR title & body

---

# 🧪 Adding New Tests

If you create or modify functionality:

- Add tests under `tests/`
- Follow naming convention:
  - `test_<module>.py`
  - `test_<component>_<behavior>.py`
- Provide:
  - happy path
  - failure modes
  - edge cases
  - stress tests (when applicable)

---

# 🔐 Security Policy

- No API keys hardcoded in codebase.
- Sensitive configs go in environment variables.
- Never commit `.env`, `*.db`, logs, or PDFs.

---

# 🌟 Contribution Types

You can contribute via:

### 📘 Code
- new agents
- optimizations
- new vector store backends
- model improvements
- better chunking
- smarter PDF parsing

### 🧪 Testing
- integration tests
- latency / performance tests
- accuracy benchmarks

### 📚 Documentation
- tutorials
- architecture diagrams
- examples & walkthroughs

### 🎓 Research contributions
- retrieval improvements
- summarization enhancements
- evaluation frameworks

---

# ❤️ Thank You

Whether you fix a typo, design a new agent, or run performance benchmarks —
your contribution matters and is deeply appreciated.

Together we make **SERA** a world-class research automation system.

```