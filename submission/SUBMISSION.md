# Submission

Fill this in and commit it. A submission missing the video link is incomplete.

## Who

- **Name:** Sarah Eaken
- **NetID:** sarahe2

## Video

Under 3 minutes, narrated, showing a cold-start pipeline run and your agent answering
a question.

Upload to Illinois MediaSpace: https://mediaspace.illinois.edu/upload/media

Set visibility to **Unlisted**.

- **Link:** https://mediaspace.illinois.edu/media/t/1_ru8kxsy2

## Chapters attempted

Mark what you completed. Partial work still gets read.

- [x] 1 · Source
- [x] 2 · Interrogate
- [x] 3 · Structure
- [x] 4 · Serve
- [x] 5 · Show
- [ ] Bonus 1 — Notice attribution
- [ ] Bonus 2 — CUSIP validation

## Checklist

- [ ] `python check_submission.py` passes
- [ ] Repo is **private** and `dsrsBOT` is a collaborator with Read access
- [x] Video uploaded to MediaSpace, visibility **Unlisted**, link tested
- [ ] Repository URL submitted at https://ikompete.dsrs.illinois.edu/competition/16
- [x] `python verify.py` passes
- [x] Pipeline run twice; output is byte-identical
- [x] `output/filings.parquet`, `output/holdings.parquet` committed
- [x] `output/filers.csv`, `output/filings/`, `submission/eda.py` committed
- [ ] `DEPENDENCIES.md`, `AI_USAGE.md`, and `ASSUMPTIONS.md` filled in
- [x] No API keys, tokens, or credentials committed
- [ ] Frozen files unmodified

## Anything we should know

The natural-language agent uses the LLM only to translate a researcher's question into
a constrained query plan. The plan is validated before execution, and retrieval and
calculations are performed deterministically in Python against read-only Parquet data.
This design prioritizes reproducibility, security, and traceable source attribution over
allowing the model to directly generate and execute arbitrary queries.

When a question does not have a unique answer or falls outside the supported dataset,
the agent returns null rather than selecting an arbitrary result or inferring unsupported
information.

With more time, I would expand the set of supported natural-language research operations
and implement notice attribution to connect 13F-NT filings to the filings in which those
managers' holdings are actually reported.

## Video sharing

We may share your video publicly to show what candidates built. If you would rather we
did not, write "do not share" here:

- **Preference:** 