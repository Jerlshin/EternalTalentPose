# `configs/lexicon/` — Lexicon Seed

[`lexicon.seed.yaml`](lexicon.seed.yaml) is the human-authored starting vocabulary for each domain concept the system needs to recognize in free-text role descriptions. It exists so competency matching depends on **meaning**, not on a candidate's exact choice of keyword.

## Current concepts

| Concept | Seed terms |
|---|---|
| `retrieval` | retrieval, dense retrieval, vector search, ann, faiss |
| `ranking` | ranking, learning to rank, ltr, reranking, ndcg |
| `recsys` | recommendation, recsys, collaborative filtering, candidate generation |
| `nlp` | nlp, natural language processing, tokenization, transformer |
| `llm` | llm, large language model, rag, prompt engineering, fine-tuning |
| `mlops` | mlops, model serving, feature store, model monitoring |

## Pipeline

1. Offline stage **O4 (Lexicon Discovery)** mines these seed terms against the candidate pool's normalized role descriptions (TF-IDF / phrase mining) to discover the full term and phrase graph for each concept.
2. Offline stage **O5 (Vocabulary Expansion)** expands each concept with embedding-nearest synonyms — terms a keyword-stuffer wouldn't anticipate adding to their profile.
3. The result is compiled to `artifacts/lexicon/lexicon.compiled.json` and `artifacts/concepts.json`, which is what `LexiconEngine` and the competency feature extractors (`src/redstack/features/skills.py`) actually consume online.

This file is never read by the online ranking run directly. See [`/ARCHITECTURE.md` §1](../../ARCHITECTURE.md#1-what-redstack-does) (resistance to keyword gaming) and [`/ARCHITECTURE.md` §6](../../ARCHITECTURE.md#6-the-engines) (`LexiconEngine`).
