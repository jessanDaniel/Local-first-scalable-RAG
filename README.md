# RAG Document Q&A — Investigation Project

An experimental, stage-by-stage investigation of Retrieval-Augmented Generation, built to
understand — not hide — every step: ingestion → chunking → embeddings → vector search →
retrieval → context construction → LLM generation → evaluation.

See [`AGENTS.md`](./AGENTS.md) for the full project brief, build order, constraints, and
learning objectives. This README documents the *results* as phases are completed; it starts
mostly empty and grows alongside the implementation.

## Status

Project scaffolded. Phase 1 (document ingestion) in progress. See `AGENTS.md` §7 for the full
phase list.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and, for later phases, [Ollama](https://ollama.com/)
running locally.

```bash
uv sync
uv run python src/main.py
```

Copy `.env.example` to `.env` and adjust as needed (defaults to local Ollama, no paid APIs).

## Running tests

```bash
uv run pytest
```

## Architecture

_To be documented as each phase lands._

## Experiments & results

_To be documented as each phase's experiments are run (chunk size/overlap, Top-K, embedding
behavior, local vs hosted LLM, retrieval/generation failure analysis)._
