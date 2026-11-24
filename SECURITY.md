# Security Policy

SERA is under development. The current ingestion baseline has known correctness and persistence limitations; see the [README](README.md#known-foundation-limitations). No production security certification or release-support guarantee is asserted.

## Reporting a vulnerability

Do not disclose sensitive vulnerability details in a public issue. Contact the repository maintainer privately to arrange a secure reporting channel. A dedicated reporting address and response-time commitments have not been established in this repository.

Include the affected revision, reproduction steps, potential impact and environment details. Redact credentials, personal data and other sensitive material. For non-sensitive correctness issues, use the repository's GitHub Issues tab.

## Security scope

The ingestion pipeline handles external network responses and untrusted PDFs, model workloads, vector storage and SQLite metadata. The approved [design and execution plan](docs/design_doc.md) specifies the planned agent security boundaries, including untrusted retrieved content, secret handling, resource limits, sandboxed code execution and fault-tolerance testing. These future controls must not be treated as implemented or validated before their applicable gates pass.

Coordinate disclosure with the maintainer before publishing details.
