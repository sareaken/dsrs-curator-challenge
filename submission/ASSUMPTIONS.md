# Assumptions

Where the specification was ambiguous, or where you asked a question and kept working
rather than waiting on an answer, record the call you made and why.

This is not a penalty. A documented assumption is a normal part of data work — the
alternative is a stalled pipeline or a silent guess nobody can audit later. We read
this alongside your output, and a well-reasoned assumption that differs from ours costs
you nothing.

| # | What was unclear | What you assumed | Why |
|---|---|---|---|
| 1 | How to handle natural-language manager names that omit legal suffixes or punctuation | Resolve manager names against the roster using normalized exact matching first, then a unique partial match; ambiguous matches return null | Researchers may write "Pershing Square Capital Management" instead of "Pershing Square Capital Management L.P." Exact matching would reject a clearly identifiable manager, while unrestricted fuzzy matching could select the wrong one. |
| 2 | How to handle a question that appears to expect one manager when multiple managers satisfy it | Return null rather than arbitrarily selecting one result | The contract prioritizes an honest inability to answer over a confident guess. For example, multiple managers held Tesla in both quarters, so the example question does not have one unique defensible manager. |
| 3 | How to interpret explicit share-versus-value wording in change questions | Explicit units in the user's question take precedence over model inference | "Shares" maps to `ssh_prnamt` where `ssh_prnamt_type == "SH"`; allowing the model to reinterpret an explicit unit as USD could produce a valid-looking but semantically incorrect answer. |
| 4 | How much authority to give the LLM over data access and calculations | Use the LLM only to translate natural language into a constrained query plan; validate the plan and perform all retrieval and calculations deterministically in Python | This keeps the Parquet data read-only, prevents model-generated executable queries or paths, improves reproducibility, and makes source attribution deterministic. |
| 5 | How to handle questions outside the supported quarters or implemented query types | Return `{"answer": null, "unit": "NONE", "sources": []}` and write the reason to stderr | The specification explicitly prefers a traceable null over an invented answer when the dataset cannot support the question. |
| 6 | How to handle local Ollama responses that do not honor vLLM's guided JSON behavior | Use a bounded JSON-extraction fallback locally, then apply the same semantic validation to the resulting query plan | The grading environment supports guided decoding, but the local Ollama model returned fenced JSON. The fallback allows local development without weakening validation or changing the frozen `agents/llm.py`. |


## Questions you sent us

If you emailed dsrs@business.illinois.edu and proceeded before hearing back, note it
here so we can see what you were working around.

| Question | Date sent | What you did in the meantime |
|---|---|---|
| | | |
