# TechSight — Design Decisions

This document records important architectural and design choices made
during development, along with the reasoning behind each decision.

---

## DD-001: Patent and Research Paper Linkage

**Decision:** We NEVER claim a patent and a research paper describe the same
invention just because their titles or keywords look similar.

**Reason:** Patent claims and research abstracts serve fundamentally different
purposes. A patent about "solid-state electrolyte" and a paper about
"solid-state electrolyte" may cover entirely different inventions, methods,
or materials.

**What we do instead:** We connect patents and papers ONLY at the
*technology-topic and year aggregation level*. Both a patent and a paper
can belong to the same KMeans topic cluster, and we count them separately
per year within that topic. We never create a direct patent↔paper link.

---

## DD-002: Data Source Selection

**OpenAlex** for research publications:
- Free API with generous rate limits
- Structured metadata including reconstructible abstracts
- Cursor pagination for large result sets

**USPTO Open Data Portal** for patents:
- Official replacement for the legacy PatentsView API (deprecated March 2026)
- Requires USPTO account with MFA + ODP API key
- Fallback: ODP bulk data download if API key is unavailable

---

## DD-003: No ORM

**Decision:** Use plain SQL with psycopg2 and parameterized queries.

**Reason:** We must understand every database operation for our viva.
An ORM hides the SQL and makes debugging harder for students. Plain SQL
is transparent and maps directly to our schema.sql.

---

## DD-004: Citation Count Handling

**Decision:** Store unavailable citation counts as NULL, never as zero.

**Reason:** Zero citations and "unknown citations" mean different things.
A patent with no citation data is not the same as a patent with zero
citations. NULL preserves this distinction.

---

*Add more decisions as the project evolves.*
