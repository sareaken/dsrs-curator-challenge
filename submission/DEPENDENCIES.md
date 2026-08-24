# Dependencies

Every library you added to `requirements-extra.txt`, with a one-line reason.

We are not counting libraries — a well-chosen dependency is better engineering than a
hand-rolled version of the same thing. What we are reading is whether you added each
one deliberately.

| Library | Version | Why |
|---|---|---|
| None | — | No additional dependencies were required beyond those provided by the project. |

## Anything you considered and rejected

No additional libraries were necessary. I used the provided project dependencies and
standard-library functionality rather than adding an end-to-end 13F retrieval or parsing
library, which kept the ingestion and parsing behavior explicit and auditable.

## Note

Libraries that wrap 13F retrieval and parsing end to end will not, on their own,
satisfy the schema or the quality report, and we will ask you to explain the edge cases
in your output regardless of how you produced it. If you can explain it, you own it.