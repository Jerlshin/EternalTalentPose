# `configs/anchors/` — Job-Description Semantic Anchors

[`jd_anchors.yaml`](jd_anchors.yaml) is the human-authored expression of "what this job description wants" and "what it doesn't," as a set of short text descriptions tagged positive or negative. This file is **re-authored per job description** — it is the one place a new role's intent enters the system.

## Current anchors

| Anchor id | Polarity | Intent |
|---|---|---|
| `jd.retrieval_ranking` | positive | Production retrieval/ranking systems with dense embeddings and learning-to-rank |
| `jd.production_ml` | positive | Shipping and operating ML models in production with monitoring and evaluation |
| `jd.product_company` | positive | Product-company engineering with end-to-end ownership, not client delivery |
| `jd.keyword_only` | negative | Tooling/keyword lists with no corroborating evidence — the anti-keyword-stuffing anchor |
| `jd.consulting_only` | negative | Services/consulting delivery with no product ownership or production exposure |
| `jd.pure_researcher` | negative | Academic research with publications but no production engineering responsibility |

## Pipeline

Offline stage **O6 (JD Concept Extraction)** embeds each anchor's text (via the same sentence-transformers model used for candidate vectors) and writes the result to `artifacts/embeddings/anchor_vectors.npy`, keyed by anchor id. Online, `SemanticEngine` (stage R3) computes each candidate's cosine similarity against every positive and negative anchor — this is the dense, semantic counterpart to the lexical anti-stuffing defense in [`../lexicon/`](../lexicon/README.md): a candidate can't talk their way past a negative anchor by avoiding its literal keywords, because the anchor match is on meaning, not vocabulary.

See [`/ARCHITECTURE.md` §6](../../ARCHITECTURE.md#6-the-engines) (`SemanticEngine`) and [`/ARCHITECTURE.md` §1](../../ARCHITECTURE.md#1-what-redstack-does).
