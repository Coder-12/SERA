# Security Policy

## Supported Versions

The SERA project follows a **rolling-release** model during early development.
At present, **only the latest commit on the `main` branch** is actively supported.

| Version     | Supported |
|-------------|-----------|
| 0.1.x (Research Agent v0) | ✔ Active |
| < 0.1.0      | ✘ Unsupported |

This will be expanded when stable releases become available.

---

## Reporting a Vulnerability

We take security seriously.  
If you discover a vulnerability, **please do *not* open a public issue.**

Instead, send a detailed report to:

📧 **security@sera-research.dev**  
*(If this email is not configured yet, replace with your preferred private email and update later.)*

Your report should include:

- A detailed description of the vulnerability  
- Steps to reproduce the issue  
- Potential impact  
- Any suggested fixes (optional but appreciated)  
- Environment details (OS, Python version, etc.)

You will receive an acknowledgment within **48 hours**, and a full response within **7 days**.

---

## Scope of Security Considerations

The following components of SERA may handle external data and therefore require careful review:

- **ArXiv Retriever** — external network calls  
- **PDF Downloader** — file streaming, network I/O  
- **PDF Parser** — handling untrusted PDFs through PyMuPDF  
- **Chunker / Embedder** — resource-heavy CPU/GPU workloads  
- **Chroma Vector Store** — interaction with local or remote DB  
- **Metadata SQLite DB** — local persistence, schema integrity  
- **Multi-Agent Pipeline** (future versions)  
- **OpenAI / API integrations** (optional)

If you believe any of these could be exploited — please contact us immediately.

---

## Coordinated Disclosure

We follow responsible disclosure practices:

1. We will investigate privately.  
2. A fix will be developed in a private branch.  
3. A security patch release will be published.  
4. You will be credited (unless you request anonymity).  
5. We will publish a security advisory only after confirmation.

---

## Non-Security Issues

For non-security bugs (performance, correctness, UX),  
please use the **GitHub Issues** tab normally.

---

## Future Security Roadmap

The following items are planned for future versions:

- Signed releases & package integrity verification  
- Optional sandboxing for PDF parsing  
- Hardened network timeouts & retries  
- Rate-limiting for internal agents  
- Encrypted vector store modes  
- Secure multi-agent communication channels  
- Secrets management (Vault / GCP Secret Manager)  
- Zero-trust module boundaries for agent orchestration  

---

Maintaining a secure and trustworthy research agent is a top priority.  
Thank you for helping us keep SERA safe and robust.
