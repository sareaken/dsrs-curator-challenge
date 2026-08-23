"""Your agent: a question in, a structured answer out.

Yours to rewrite. One thing is fixed — `main(question) -> dict` must exist, because we
call it directly:

    python -m agents.answer "which manager held the largest Apple position in 2026 Q2?"

Return this shape. Nothing else on stdout.

    {
      "answer":  <number | string | list | null>,
      "unit":    "USD" | "SHARES" | "COUNT" | "PERCENT" | "NAME" | "DATE" | "NONE",
      "sources": ["0001423053-26-000012", ...]
    }

`sources` is graded separately from `answer`, and it is the more diagnostic of the two.
An agent that produces the right number without knowing which filings it came from is
not one a researcher can trust with a question they cannot check by hand.

Put logging on stderr. stdout carries the JSON and nothing else.

See agents/llm.py for the model interface, and docs/04-serve.md for what is graded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import re
import pyarrow.parquet as pq
import os
from agents.llm import complete, complete_json, LLMError

OUTPUT = Path(__file__).resolve().parents[1] / "output"
FILINGS = OUTPUT / "filings.parquet"
HOLDINGS = OUTPUT / "holdings.parquet"

VALID_UNITS = {"USD", "SHARES", "COUNT", "PERCENT", "NAME", "DATE", "NONE"}


QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": [
                "largest_position",
                "largest_change",
                "distinct_issuers",
                "total_value",
                "form_managers",
                "has_position",
                "most_options",
                "largest_fund_position",
                "issuer_change",
                "unsupported",
            ],
        },
        "fund_name": {"type": ["string", "null"]},
        "issuer": {"type": ["string", "null"]},
        "quarter": {"type": ["string", "null"]},
        "comparison_quarter": {"type": ["string", "null"]},
        "metric": {
            "type": ["string", "null"],
            "enum": ["value", "shares", "count", None],
        },
        "option_type": {
            "type": ["string", "null"],
            "enum": ["call", "put", None],
        },
        "form_type": {
            "type": ["string", "null"],
            "enum": ["13F-HR", "13F-NT", None],
        },
    },
    "required": [
        "operation",
        "fund_name",
        "issuer",
        "quarter",
        "comparison_quarter",
        "metric",
        "option_type",
        "form_type",
    ],
}


SYSTEM_PROMPT = """
Translate the user's question about SEC Form 13F data into a constrained query plan.

Return ONLY a single JSON object.
Do not use Markdown or code fences.
Do not include commentary before or after the JSON.
The JSON must be a FLAT object. Never create nested objects such as "filters".

Use exactly these fields:
- operation
- fund_name
- issuer
- quarter
- comparison_quarter
- metric
- option_type
- form_type

Every field must be present.
Use null for fields that do not apply.

The dataset contains only 2026Q1 and 2026Q2.

Valid operations and when to use them:
- largest_position:
  Use ONLY when comparing multiple managers to find which manager has the
  largest position in a SPECIFIC issuer.
  Requires issuer and quarter.
- largest_change:
  Use when comparing multiple managers to find who increased or decreased
  a SPECIFIC issuer the most between two quarters.
  Requires issuer, quarter, comparison_quarter, and metric.
- distinct_issuers:
  Use when counting distinct issuers for ONE specified manager.
  Requires fund_name and quarter.
- total_value:
  Use when asking for the total portfolio/holdings value of ONE specified manager.
  Requires fund_name and quarter.
- form_managers:
  Use ONLY when asking which managers filed a particular SEC form type,
  such as 13F-HR or 13F-NT.
  Requires form_type and quarter.
- has_position:
  Use when asking whether ONE specified manager reported a position in a
  SPECIFIC issuer.
  Requires fund_name, issuer, and quarter.
- most_options:
  Use when asking which manager reported the most call or put option positions.
  Requires option_type and quarter.
- largest_fund_position:
  Use when asking what the largest position was for ONE specified manager.
  This does NOT require an issuer because the question is asking you to find
  the issuer.
  Requires fund_name and quarter.
- issuer_change:
  Use when asking about holding a SPECIFIC issuer in BOTH quarters and whether
  that position grew or shrank.
  Requires issuer, quarter, and comparison_quarter.
- unsupported:
  Use when the requested question cannot be answered from this dataset.

Important distinctions:
- "Which manager had the largest Apple position?" = largest_position.
- "What was Third Point LLC's largest position?" = largest_fund_position.
- "Did Pershing Square hold Microsoft?" = has_position.
- "Who filed 13F-NT?" = form_managers.
- "Who held Tesla in both quarters and did it grow or shrink?" = issuer_change.
- Words such as "manager" do NOT imply form_managers. form_managers is ONLY
  for questions explicitly about SEC form types.

Manager names:
- Preserve the manager name exactly as written in the user's question.
- Do not invent or remove legal suffixes.
- A partial manager name may be provided by the user; data access code must
  resolve it safely rather than assuming an exact match.

Rules:
- Use unsupported for questions outside the available quarters.
- "shares" means ssh_prnamt where ssh_prnamt_type == SH.
- Position value means filed USD value.
- Preserve manager and issuer text from the question.
- Never invent data.
- Never calculate the final answer yourself.
- Your only job is to classify the question and extract its parameters.
- comparison_quarter is the earlier/baseline quarter.
- quarter is the later/target quarter.
- fund_name is only for a manager explicitly named in the question.
- If no manager is explicitly named, fund_name must be null.
- issuer is the company/security named in the question, such as Apple, Nvidia, Tesla, or Microsoft.

Example:

{
  "operation": "largest_position",
  "fund_name": null,
  "issuer": "Apple",
  "quarter": "2026Q2",
  "comparison_quarter": null,
  "metric": "value",
  "option_type": null,
  "form_type": null
}
"""


def normalize_text(value: str) -> str:
    value = value.upper()
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def resolve_fund_name(
    filings: list[dict],
    query_name: str,
) -> str | None:
    """Resolve a user-supplied manager name to exactly one roster fund name."""

    query = normalize_text(query_name)

    roster_names = sorted(
        {
            row["fund_name"]
            for row in filings
        }
    )

    exact_matches = [
        name
        for name in roster_names
        if normalize_text(name) == query
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    partial_matches = [
        name
        for name in roster_names
        if (
            normalize_text(name).startswith(query)
            or query.startswith(normalize_text(name))
        )
    ]

    if len(partial_matches) == 1:
        return partial_matches[0]

    if len(partial_matches) > 1:
        print(
            f"Ambiguous fund name {query_name!r}: {partial_matches}",
            file=sys.stderr,
        )

    return None


def load_data() -> tuple[list[dict], list[dict]]:
    if not FILINGS.exists() or not HOLDINGS.exists():
        raise FileNotFoundError(
            "Required Parquet files are missing. Run main.py first."
        )

    filings = pq.read_table(FILINGS).to_pylist()
    holdings = pq.read_table(HOLDINGS).to_pylist()

    return filings, holdings


def parse_question(question: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        return complete_json(
            messages,
            QUERY_SCHEMA,
            max_tokens=600,
        )

    except LLMError:
        # Local OpenAI-compatible servers such as Ollama may ignore
        # vLLM's guided_json parameter and return fenced JSON.
        raw = complete(
            messages,
            max_tokens=600,
        ).strip()

        if raw.startswith("```"):
            raw = re.sub(
                r"^```(?:json)?\s*",
                "",
                raw,
                flags=re.IGNORECASE,
            )
            raw = re.sub(
                r"\s*```$",
                "",
                raw,
            )

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Model did not return usable JSON: {raw[:300]!r}"
            ) from exc

        return plan


def null_answer(reason: str) -> dict[str, Any]:
    print(reason, file=sys.stderr)

    return {
        "answer": None,
        "unit": "NONE",
        "sources": [],
    }


def issuer_matches(raw_issuer: str, query_issuer: str) -> bool:
    """Cautiously match a user's issuer text to filed issuer text."""

    raw = normalize_text(raw_issuer)
    query = normalize_text(query_issuer)

    if not raw or not query:
        return False

    return (
        raw == query
        or raw.startswith(query + " ")
        or query in raw.split()
    )


def largest_position(
    filings: list[dict],
    holdings: list[dict],
    issuer: str,
    quarter: str,
) -> dict[str, Any]:
    """Return the manager with the largest reported USD position."""

    filing_by_accession = {
        row["accession_number"]: row
        for row in filings
    }

    totals: dict[str, int] = {}
    sources: dict[str, set[str]] = {}

    for row in holdings:
        if row["report_quarter"] != quarter:
            continue

        if not issuer_matches(
            row["name_of_issuer"],
            issuer,
        ):
            continue

        accession = row["accession_number"]
        filing = filing_by_accession.get(accession)

        if filing is None:
            continue

        fund_name = filing["fund_name"]

        totals[fund_name] = (
            totals.get(fund_name, 0)
            + row["value"]
        )

        sources.setdefault(
            fund_name,
            set(),
        ).add(accession)

    if not totals:
        return null_answer(
            f"No holdings matched issuer={issuer!r} "
            f"in quarter={quarter!r}."
        )

    winner = unique_max(
        totals,
        description=f"largest {issuer} position in {quarter}",
    )

    if winner is None:
        return null_answer(
            f"No unique largest {issuer!r} position exists in {quarter}."
        )

    fund_name, value = winner

    return {
        "answer": value,
        "unit": "USD",
        "sources": sorted(sources[fund_name]),
    }


def get_filing_lookup(
    filings: list[dict],
) -> dict[str, dict]:
    return {
        row["accession_number"]: row
        for row in filings
    }


def largest_change(
    filings: list[dict],
    holdings: list[dict],
    issuer: str,
    quarter: str,
    comparison_quarter: str,
    metric: str,
) -> dict[str, Any]:
    """Find the manager with the largest increase between two quarters."""

    filing_by_accession = get_filing_lookup(filings)

    totals: dict[tuple[str, str], int] = {}
    sources: dict[tuple[str, str], set[str]] = {}

    for row in holdings:
        if row["report_quarter"] not in {
            quarter,
            comparison_quarter,
        }:
            continue

        if not issuer_matches(
            row["name_of_issuer"],
            issuer,
        ):
            continue

        if metric == "shares":
            if row["ssh_prnamt_type"] != "SH":
                continue
            amount = row["ssh_prnamt"]

        elif metric == "value":
            amount = row["value"]

        else:
            return null_answer(
                f"Unsupported metric for largest_change: {metric!r}"
            )

        accession = row["accession_number"]
        filing = filing_by_accession.get(accession)

        if filing is None:
            continue

        fund_name = filing["fund_name"]
        key = (fund_name, row["report_quarter"])

        totals[key] = totals.get(key, 0) + amount
        sources.setdefault(key, set()).add(accession)

    managers = {
        fund_name
        for fund_name, _ in totals
    }

    changes = []

    for fund_name in managers:
        newer = totals.get(
            (fund_name, quarter),
            0,
        )
        older = totals.get(
            (fund_name, comparison_quarter),
            0,
        )

        change = newer - older

        changes.append(
            (
                fund_name,
                change,
            )
        )

    if not changes:
        return null_answer(
            f"No matching holdings found for issuer={issuer!r}."
        )

    change_lookup = {
        fund_name: change
        for fund_name, change in changes
    }

    winner = unique_max(
        change_lookup,
        description=(
            f"largest change in {issuer} "
            f"from {comparison_quarter} to {quarter}"
        ),
    )

    if winner is None:
        return null_answer(
            f"No unique manager had the largest change "
            f"in {issuer!r}."
        )

    fund_name, change = winner

    source_list = filing_sources_for_manager(
        filings,
        fund_name,
        {
            quarter,
            comparison_quarter,
        },
    )

    return {
        "answer": change,
        "unit": (
            "SHARES"
            if metric == "shares"
            else "USD"
        ),
        "sources": source_list,
    }


def distinct_issuers(
    filings: list[dict],
    holdings: list[dict],
    fund_name: str,
    quarter: str,
) -> dict[str, Any]:
    """Count distinct issuer names for one manager and quarter."""

    filing_by_accession = get_filing_lookup(filings)

    issuers = set()
    source_set = set()

    resolved_fund = resolve_fund_name(
        filings,
        fund_name,
    )

    if resolved_fund is None:
        return null_answer(
            f"Could not uniquely resolve fund name {fund_name!r}."
        )

    normalized_fund = normalize_text(resolved_fund)

    for row in holdings:
        if row["report_quarter"] != quarter:
            continue

        filing = filing_by_accession.get(
            row["accession_number"]
        )

        if filing is None:
            continue

        if normalize_text(filing["fund_name"]) != normalized_fund:
            continue

        issuers.add(
            normalize_text(row["name_of_issuer"])
        )
        source_set.add(
            row["accession_number"]
        )

    if not source_set:
        return null_answer(
            f"No filing found for {fund_name!r} in {quarter}."
        )

    return {
        "answer": len(issuers),
        "unit": "COUNT",
        "sources": sorted(source_set),
    }


def total_value(
    filings: list[dict],
    fund_name: str,
    quarter: str,
) -> dict[str, Any]:
    """Return declared portfolio value for one manager and quarter."""

    resolved_fund = resolve_fund_name(
        filings,
        fund_name,
    )

    if resolved_fund is None:
        return null_answer(
            f"Could not uniquely resolve fund name {fund_name!r}."
        )

    normalized_fund = normalize_text(resolved_fund)

    matches = [
        row
        for row in filings
        if normalize_text(row["fund_name"]) == normalized_fund
        and row["report_quarter"] == quarter
    ]

    if not matches:
        return null_answer(
            f"No filing found for {fund_name!r} in {quarter}."
        )

    if len(matches) != 1:
        return null_answer(
            f"Expected one filing for {fund_name!r} in {quarter}, "
            f"found {len(matches)}."
        )

    filing = matches[0]

    if filing["table_value_total"] is None:
        return null_answer(
            f"{fund_name!r} has no holdings table in {quarter}."
        )

    return {
        "answer": filing["table_value_total"],
        "unit": "USD",
        "sources": [filing["accession_number"]],
    }


def form_managers(
    filings: list[dict],
    quarter: str,
    form_type: str,
) -> dict[str, Any]:
    """List roster managers that filed a given form in one quarter."""

    matches = [
        row
        for row in filings
        if row["report_quarter"] == quarter
        and row["form_type"] == form_type
    ]

    managers = sorted(
        {
            row["fund_name"]
            for row in matches
        }
    )

    if not managers:
        return {
            "answer": [],
            "unit": "NAME",
            "sources": [],
        }

    return {
        "answer": managers,
        "unit": "NAME",
        "sources": sorted(
            row["accession_number"]
            for row in matches
        ),
    }


def has_position(
    filings: list[dict],
    holdings: list[dict],
    fund_name: str,
    issuer: str,
    quarter: str,
) -> dict[str, Any]:
    """Return whether a manager directly reported any matching holding."""

    filing_by_accession = get_filing_lookup(filings)

    resolved_fund = resolve_fund_name(
        filings,
        fund_name,
    )

    if resolved_fund is None:
        return null_answer(
            f"Could not uniquely resolve fund name {fund_name!r}."
        )

    normalized_fund = normalize_text(resolved_fund)

    manager_filings = [
        row
        for row in filings
        if normalize_text(row["fund_name"]) == normalized_fund
        and row["report_quarter"] == quarter
    ]

    if not manager_filings:
        return null_answer(
            f"No filing found for {resolved_fund!r} in {quarter}."
        )

    matching_sources = set()

    for row in holdings:
        if row["report_quarter"] != quarter:
            continue

        filing = filing_by_accession.get(
            row["accession_number"]
        )

        if filing is None:
            continue

        if normalize_text(filing["fund_name"]) != normalized_fund:
            continue

        if issuer_matches(
            row["name_of_issuer"],
            issuer,
        ):
            matching_sources.add(
                row["accession_number"]
            )

    if matching_sources:
        return {
            "answer": "YES",
            "unit": "NONE",
            "sources": sorted(matching_sources),
        }

    return {
        "answer": "NO",
        "unit": "NONE",
        "sources": sorted(
            row["accession_number"]
            for row in manager_filings
        ),
    }


def most_options(
    filings: list[dict],
    holdings: list[dict],
    quarter: str,
    option_type: str,
) -> dict[str, Any]:
    """Find manager with the largest count of option position rows."""

    filing_by_accession = get_filing_lookup(filings)

    counts: dict[str, int] = {}
    sources: dict[str, set[str]] = {}

    target = option_type.lower()

    for row in holdings:
        if row["report_quarter"] != quarter:
            continue

        put_call = row["put_call"]

        if put_call is None:
            continue

        if put_call.lower() != target:
            continue

        filing = filing_by_accession.get(
            row["accession_number"]
        )

        if filing is None:
            continue

        fund_name = filing["fund_name"]

        counts[fund_name] = (
            counts.get(fund_name, 0) + 1
        )

        sources.setdefault(
            fund_name,
            set(),
        ).add(row["accession_number"])

    if not counts:
        return null_answer(
            f"No {option_type} option positions found in {quarter}."
        )

    winner = unique_max(
        counts,
        description=f"most {option_type} options in {quarter}",
    )

    if winner is None:
        return null_answer(
            f"No unique manager reported the most "
            f"{option_type} options in {quarter}."
        )

    fund_name, count = winner

    return {
        "answer": fund_name,
        "unit": "NAME",
        "sources": sorted(sources[fund_name]),
    }


def largest_fund_position(
    filings: list[dict],
    holdings: list[dict],
    fund_name: str,
    quarter: str,
) -> dict[str, Any]:
    """Return one manager's largest issuer position by filed USD value."""

    filing_by_accession = get_filing_lookup(filings)

    resolved_fund = resolve_fund_name(
        filings,
        fund_name,
    )

    if resolved_fund is None:
        return null_answer(
            f"Could not uniquely resolve fund name {fund_name!r}."
        )

    normalized_fund = normalize_text(resolved_fund)

    by_issuer: dict[str, int] = {}
    display_name: dict[str, str] = {}
    sources: dict[str, set[str]] = {}

    for row in holdings:
        if row["report_quarter"] != quarter:
            continue

        filing = filing_by_accession.get(
            row["accession_number"]
        )

        if filing is None:
            continue

        if normalize_text(filing["fund_name"]) != normalized_fund:
            continue

        issuer_key = normalize_text(
            row["name_of_issuer"]
        )

        by_issuer[issuer_key] = (
            by_issuer.get(issuer_key, 0)
            + row["value"]
        )

        display_name.setdefault(
            issuer_key,
            row["name_of_issuer"],
        )

        sources.setdefault(
            issuer_key,
            set(),
        ).add(row["accession_number"])

    if not by_issuer:
        return null_answer(
            f"No holdings found for {fund_name!r} in {quarter}."
        )

    winner = unique_max(
        by_issuer,
        description=(
            f"largest position for {fund_name} in {quarter}"
        ),
    )

    if winner is None:
        return null_answer(
            f"No unique largest position exists for "
            f"{fund_name!r} in {quarter}."
        )

    issuer_key, value = winner

    return {
        "answer": [
            value,
            display_name[issuer_key],
        ],
        "unit": "NONE",
        "sources": sorted(sources[issuer_key]),
    }


def issuer_change(
    filings: list[dict],
    holdings: list[dict],
    issuer: str,
    quarter: str,
    comparison_quarter: str,
) -> dict[str, Any]:
    """Return the unique manager holding an issuer in both quarters and direction."""

    filing_by_accession = get_filing_lookup(filings)

    values: dict[tuple[str, str], int] = {}
    sources: dict[str, set[str]] = {}

    for row in holdings:
        if row["report_quarter"] not in {
            quarter,
            comparison_quarter,
        }:
            continue

        if not issuer_matches(
            row["name_of_issuer"],
            issuer,
        ):
            continue

        filing = filing_by_accession.get(
            row["accession_number"]
        )

        if filing is None:
            continue

        fund_name = filing["fund_name"]

        key = (
            fund_name,
            row["report_quarter"],
        )

        values[key] = (
            values.get(key, 0)
            + row["value"]
        )

        sources.setdefault(
            fund_name,
            set(),
        ).add(row["accession_number"])

    managers = sorted(
        {
            fund
            for fund, qtr in values
            if (
                (fund, quarter) in values
                and (fund, comparison_quarter) in values
            )
        }
    )

    if not managers:
        return null_answer(
            f"No manager held {issuer!r} in both requested quarters."
        )

    if len(managers) != 1:
        return null_answer(
            f"Question does not have a unique answer: "
            f"{len(managers)} managers held {issuer!r} "
            f"in both {comparison_quarter} and {quarter}."
        )

    fund_name = managers[0]

    newer = values[
        (fund_name, quarter)
    ]
    older = values[
        (fund_name, comparison_quarter)
    ]

    if newer > older:
        direction = "grew"
    elif newer < older:
        direction = "shrank"
    else:
        direction = "unchanged"

    return {
        "answer": [
            fund_name,
            direction,
        ],
        "unit": "NONE",
        "sources": filing_sources_for_manager(
            filings,
            fund_name,
            {
                quarter,
                comparison_quarter,
            },
        ),
    }


def unique_max(
    values: dict[str, int],
    *,
    description: str,
) -> tuple[str, int] | None:
    """Return a unique maximum, or None when the top value is tied."""

    if not values:
        return None

    max_value = max(values.values())

    winners = sorted(
        name
        for name, value in values.items()
        if value == max_value
    )

    if len(winners) != 1:
        print(
            f"Ambiguous {description}: "
            f"{len(winners)} entities tie at {max_value}: {winners}",
            file=sys.stderr,
        )
        return None

    return winners[0], max_value


def filing_sources_for_manager(
    filings: list[dict],
    fund_name: str,
    quarters: set[str],
) -> list[str]:
    """Return filing accessions for one manager across requested quarters."""

    normalized_fund = normalize_text(fund_name)

    return sorted(
        row["accession_number"]
        for row in filings
        if normalize_text(row["fund_name"]) == normalized_fund
        and row["report_quarter"] in quarters
    )


def repair_plan_from_question(
    question: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Apply conservative corrections when the user's wording is explicit."""

    repaired = dict(plan)
    q = question.lower()

    # Explicit units in the user's question outrank model inference.
    if repaired.get("operation") == "largest_change":
        if "share" in q:
            repaired["metric"] = "shares"
        elif "value" in q or "usd" in q or "dollar" in q:
            repaired["metric"] = "value"

    return repaired


def validate_plan(plan: dict[str, Any]) -> tuple[bool, str]:
    """Validate operation-specific query-plan requirements."""

    valid_quarters = {"2026Q1", "2026Q2"}

    operation = plan.get("operation")

    if operation == "unsupported":
        return True, ""

    if plan.get("quarter") is not None:
        if plan["quarter"] not in valid_quarters:
            return False, f"Unsupported quarter: {plan['quarter']!r}"

    if plan.get("comparison_quarter") is not None:
        if plan["comparison_quarter"] not in valid_quarters:
            return False, (
                f"Unsupported comparison quarter: "
                f"{plan['comparison_quarter']!r}"
            )

    required_fields = {
        "largest_position": [
            "issuer",
            "quarter",
        ],
        "largest_change": [
            "issuer",
            "quarter",
            "comparison_quarter",
            "metric",
        ],
        "distinct_issuers": [
            "fund_name",
            "quarter",
        ],
        "total_value": [
            "fund_name",
            "quarter",
        ],
        "form_managers": [
            "quarter",
            "form_type",
        ],
        "has_position": [
            "fund_name",
            "issuer",
            "quarter",
        ],
        "most_options": [
            "quarter",
            "option_type",
        ],
        "largest_fund_position": [
            "fund_name",
            "quarter",
        ],
        "issuer_change": [
            "issuer",
            "quarter",
            "comparison_quarter",
        ],
    }

    if operation not in required_fields:
        return False, f"Unknown operation: {operation!r}"

    missing = [
        field
        for field in required_fields[operation]
        if plan.get(field) is None
    ]

    if missing:
        return False, (
            f"Operation {operation!r} is missing required "
            f"field(s): {missing}"
        )

    if operation == "largest_change":
        if plan["quarter"] == plan["comparison_quarter"]:
            return False, "Comparison quarters must be different."

        if plan["metric"] not in {"shares", "value"}:
            return False, (
                "largest_change metric must be "
                "'shares' or 'value'."
            )

    if operation == "issuer_change":
        if plan["quarter"] == plan["comparison_quarter"]:
            return False, "Comparison quarters must be different."

    if operation == "form_managers":
        if plan["form_type"] not in {"13F-HR", "13F-NT"}:
            return False, (
                "form_managers requires form_type "
                "13F-HR or 13F-NT."
            )

    if operation == "most_options":
        if plan["option_type"] not in {"call", "put"}:
            return False, (
                "most_options requires option_type "
                "'call' or 'put'."
            )

    return True, ""


def main(question: str) -> dict[str, Any]:
    """Answer `question` against your dataset."""

    try:
        if os.environ.get("LLM_MODE", "live").lower() == "mock":
            return {
                "answer": None,
                "unit": "NONE",
                "sources": [],
            }

        plan = parse_question(question)

        plan = repair_plan_from_question(
            question,
            plan,
        )

        valid, reason = validate_plan(plan)

        if not valid:
            return null_answer(
                f"Invalid query plan: {reason}"
            )

        if plan["operation"] == "unsupported":
            return null_answer(
                "Question is outside the supported dataset or query types."
            )

        filings, holdings = load_data()

        operation = plan["operation"]

        if operation == "largest_position":
            return largest_position(
                filings,
                holdings,
                plan["issuer"],
                plan["quarter"],
            )

        if operation == "largest_change":
            return largest_change(
                filings,
                holdings,
                plan["issuer"],
                plan["quarter"],
                plan["comparison_quarter"],
                plan["metric"],
            )

        if operation == "distinct_issuers":
            return distinct_issuers(
                filings,
                holdings,
                plan["fund_name"],
                plan["quarter"],
            )

        if operation == "total_value":
            return total_value(
                filings,
                plan["fund_name"],
                plan["quarter"],
            )

        if operation == "form_managers":
            return form_managers(
                filings,
                plan["quarter"],
                plan["form_type"],
            )

        if operation == "has_position":
            return has_position(
                filings,
                holdings,
                plan["fund_name"],
                plan["issuer"],
                plan["quarter"],
            )

        if operation == "most_options":
            return most_options(
                filings,
                holdings,
                plan["quarter"],
                plan["option_type"],
            )

        if operation == "largest_fund_position":
            return largest_fund_position(
                filings,
                holdings,
                plan["fund_name"],
                plan["quarter"],
            )

        if operation == "issuer_change":
            return issuer_change(
                filings,
                holdings,
                plan["issuer"],
                plan["quarter"],
                plan["comparison_quarter"],
            )

        return null_answer(
            f"Unsupported operation: {operation!r}"
        )

    except Exception as exc:
        return null_answer(
            f"Could not answer question: {exc}"
        )


def _cli() -> int:
    if len(sys.argv) < 2:
        print('usage: python -m agents.answer "your question"', file=sys.stderr)
        return 2

    result = main(sys.argv[1])

    # Validated here so a shape mistake surfaces while you can still fix it. The grader
    # parses stdout as JSON and reads exactly these three keys.
    if not isinstance(result, dict):
        print(f"main() must return a dict, got {type(result).__name__}", file=sys.stderr)
        return 1
    missing = {"answer", "unit", "sources"} - set(result)
    if missing:
        print(f"result missing key(s): {sorted(missing)}", file=sys.stderr)
        return 1
    if result["unit"] not in VALID_UNITS:
        print(f"unit must be one of {sorted(VALID_UNITS)}, got {result['unit']!r}",
              file=sys.stderr)
        return 1
    if not isinstance(result["sources"], list):
        print("sources must be a list of accession numbers", file=sys.stderr)
        return 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
