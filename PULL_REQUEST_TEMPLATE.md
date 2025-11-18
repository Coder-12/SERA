
````markdown
## 🚀 Summary

_Provide a short summary of the changes introduced in this PR._

Examples:
- Adds batch-stable embedder retry logic for production reliability.
- Fixes ChromaAdapter include detection across versions.
- Implements Research Agent v0 ingestion pipeline (arXiv → PDF → parse → chunk → embed → store).

---

## 🧠 Motivation

_Why is this change necessary? What problem does it solve?_

Examples:
- Ensures embedding operations are resilient to transient failures.
- Enables deterministic chunk metadata linking for downstream agents.
- Implements missing API consistency across adapters.

---

## 🔍 Related Issues / Tickets

_Link any issue numbers or design documents._

- Closes #123
- Relates to design: `/docs/architecture/research_agent_v0.md`

---

## 📦 What Changed?

### **New Features**
- [ ] SentenceTransformers async-safe encoding
- [ ] Metadata DB chunk-level indexing
- [ ] Chroma API version-adaptive include logic

### **Fixes**
- [ ] Fixed dimension mismatch handling in vector adapters
- [ ] Resolved PDFParser partial-page extraction edge case
- [ ] Stabilized retry logic for `_with_retries()`

### **Refactors**
- [ ] Improved naming consistency across modules
- [ ] Extracted utility helpers into `utils/`

---

## 🧪 Testing Done

### **Unit Tests**
- [ ] Added new tests covering this feature
- [ ] All tests pass locally  
  ```bash
  pytest -q
````

### **Integration Tests**

* [ ] Verified end-to-end ingestion

  ```bash
  pytest tests/*integration* -q
  ```

### **Live Tests (optional)**

* [ ] Validated real arXiv → PDF → parse → chunk → embed → chroma store

  ```bash
  pytest tests/test_full_integration_live.py -s
  ```

Attach output if relevant.

---

## 🔬 How to Review

*Check all that apply:*

* [ ] Small building block — easy to review
* [ ] Medium-sized change — requires careful review
* [ ] Large PR — please review commit-by-commit
* [ ] Needs async / concurrency expertise
* [ ] Needs vector-store expertise
* [ ] Needs ML/NLP expertise

---

## ⚡ Breaking Changes?

Does this PR introduce breaking changes?

* [ ] Yes
* [ ] No

If yes, describe impact and migration steps:

---

## 📚 Documentation Updated?

* [ ] README updated
* [ ] Architecture docs updated
* [ ] Example scripts updated
* [ ] Inline docstrings added/updated

---

## ❤️ Thank You

Thank you for reviewing this PR! Your feedback directly improves SERA’s engineering quality and research capabilities.
