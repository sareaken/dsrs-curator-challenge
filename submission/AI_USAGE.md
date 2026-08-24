# AI Usage

Declare what you used and how. We are not scoring the amount — we are checking that
you can account for your own work.

## Tools used

- ChatGPT (OpenAI) — used throughout the project for requirement interpretation, implementation guidance, code development/debugging, validation, and explaining unfamiliar SEC/13F concepts.
- Ollama with Gemma 3 4B — used as a local OpenAI-compatible model for developing and testing the Chapter 4 natural-language agent.
- Grading model target: the agent was designed to run against the provided `agents/llm.py` interface so that the local model can be replaced by the grading environment's model without code changes.

## Where you used them

Roughly, by chapter. A sentence each is enough.

| Chapter | How you used AI |
|---|---|
| 1 · Source | Used ChatGPT to help interpret the SEC/EDGAR ingestion requirements, develop and debug the Python pipeline, troubleshoot the local environment, and validate filing discovery, caching, and deterministic output behavior. |
| 2 · Interrogate | Used ChatGPT to help design the exploratory analysis, interpret the 13F schema and XML structure, and review the evidence produced by `eda.py`, including optional fields, notice filings, CUSIP characteristics, and differences between roster and filing-manager names. |
| 3 · Structure | Used ChatGPT to help implement and debug the XML-to-Parquet parsing logic, reason through namespace handling, nulls, types, notice filings, duplicate holdings, and deterministic output, and interpret the results of the provided validator. |
| 4 · Serve | Used ChatGPT extensively to help design, implement, debug, and test the natural-language agent. The final architecture uses an LLM only to translate questions into a constrained query plan; validated deterministic Python performs retrieval and calculations. I used Gemma 3 4B through Ollama locally to test the LLM integration and used ChatGPT to diagnose model-output and answer-semantics issues. |

## What you would change

I understand the overall architecture and the major data transformations in the submitted
code, but AI assistance was used substantially during implementation and debugging. Given
more time, I would simplify and refactor parts of the Chapter 4 agent code, expand its
supported query operations, and add more automated tests around natural-language query
interpretation and answer semantics. I would also test against a model closer to the
grading model; the smaller local Gemma model did not always follow structured-output
instructions consistently, which required additional defensive handling during local
development.