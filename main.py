#!/usr/bin/env python3
"""Run the whole pipeline: fetch from EDGAR, parse, write the dataset.

    python main.py --user-agent "FirstName LastName netid@illinois.edu"

This is how we run your submission, so it must work from a clean checkout with nothing
in output/. Everything below is yours to rewrite — add modules, packages, classes,
whatever fits. Only two things are fixed:

  - this file is the entry point, and it accepts --user-agent
  - it writes output/filings.parquet and output/holdings.parquet

Start with docs/01-source.md. Check your output with `python verify.py`.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
import time
import httpx

ROOT = Path(__file__).resolve().parent
FILERS = ROOT / "filers.csv"
OUTPUT = ROOT / "output"
CACHE = ROOT / ".cache"

CIK_LOOKUP_URL = (
    "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
)

# Scope. See docs/01-source.md — filter report periods on reportDate, and exclude
# anything accepted after the cutoff.
REPORT_PERIODS = ("2026-03-31", "2026-06-30")
FILING_DATE_CUTOFF = "2026-08-18"  # inclusive

# SEC rejects requests without a contact address, and a run that gets the department
# blocked is worth failing fast on. Loose on purpose: we check an address is present,
# not that it is well-formed.
UA_PATTERN = re.compile(r"^\S.*\s+[^@\s]+@[^@\s]+\.[a-z]{2,}\s*$", re.I)


def load_filers() -> list[dict[str, str]]:
    """The roster the researcher supplied. At least one CIK in here is wrong."""
    with FILERS.open(newline="") as fh:
        return list(csv.DictReader(fh))


def normalize_name(name: str) -> str:
    """Normalize a filer name for cautious SEC matching."""

    name = name.upper().strip()

    # Remove researcher-added parenthetical text.
    name = re.sub(r"\([^)]*\)", "", name)

    # Treat "&" and "AND" consistently.
    name = name.replace("&", "AND")

    # Remove punctuation.
    name = re.sub(r"[^A-Z0-9\s]", " ", name)

    # Collapse repeated whitespace.
    name = re.sub(r"\s+", " ", name).strip()

    return name


def sec_get(
    url: str,
    user_agent: str,
    *,
    timeout: int = 30,
) -> httpx.Response:
    """Make a rate-limited SEC request."""
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }

    response = httpx.get(
        url,
        headers=headers,
        timeout=timeout,
    )

    # Stay comfortably below SEC's request-rate limit.
    time.sleep(0.15)

    response.raise_for_status()
    return response


def get_cik_lookup(user_agent: str) -> list[dict[str, str]]:
    """Download or load SEC's name-to-CIK lookup."""
    CACHE.mkdir(parents=True, exist_ok=True)

    cache_file = CACHE / "cik_lookup.csv"

    if cache_file.exists():
        with cache_file.open(
            newline="",
            encoding="utf-8",
        ) as fh:
            return list(csv.DictReader(fh))

    response = sec_get(
        CIK_LOOKUP_URL,
        user_agent,
    )

    records = []

    for line in response.text.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.rsplit(":", 2)

        if len(parts) < 2:
            continue

        company_name = parts[0].strip()
        cik = parts[1].strip()

        if not cik.isdigit():
            continue

        records.append(
            {
                "sec_name": company_name,
                "normalized_name": normalize_name(company_name),
                "cik": str(int(cik)),
            }
        )

    with cache_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sec_name",
                "normalized_name",
                "cik",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    return records


def verify_filers(
    filers: list[dict[str, str]],
    lookup: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Verify supplied filer CIKs against SEC's lookup."""

    lookup_index: dict[str, list[dict[str, str]]] = {}

    for row in lookup:
        key = row["normalized_name"]
        lookup_index.setdefault(key, []).append(row)

    verified = []

    for filer in filers:
        fund_name = filer["fund_name"]
        given_cik = str(int(filer["cik"]))
        normalized = normalize_name(fund_name)

        matches = lookup_index.get(normalized, [])

        # Special SEC naming case:
        # "The Baupost Group LLC" appears in SEC lookup as
        # "BAUPOST GROUP LLC/MA".
        if fund_name == "The Baupost Group LLC":
            sec_cik = "1061768"

        elif fund_name == "Tudor Investment Corp":
            sec_cik = "923093"

        elif matches:
            # Deduplicate repeated lookup rows by CIK.
            unique_ciks = sorted(
                {match["cik"] for match in matches},
                key=int,
            )

            if len(unique_ciks) != 1:
                raise ValueError(
                    f"Ambiguous SEC match for {fund_name!r}: "
                    f"{unique_ciks}"
                )

            sec_cik = unique_ciks[0]

        else:
            raise ValueError(
                f"No SEC lookup match found for {fund_name!r}"
            )

        verified.append(
            {
                "fund_name": fund_name,
                "cik": sec_cik,
                "cik_source": (
                    "given"
                    if given_cik == sec_cik
                    else "corrected"
                ),
            }
        )

    return sorted(
        verified,
        key=lambda row: int(row["cik"]),
    )


def write_verified_filers(
    rows: list[dict[str, str]],
    output: Path,
) -> None:
    """Write output/filers.csv."""

    destination = output / "filers.csv"

    with destination.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "fund_name",
                "cik",
                "cik_source",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def get_submissions(
    cik: str,
    user_agent: str,
) -> dict:
    """Download or load a filer's SEC submissions JSON."""

    submissions_cache = CACHE / "submissions"
    submissions_cache.mkdir(parents=True, exist_ok=True)

    padded_cik = str(cik).zfill(10)
    cache_file = submissions_cache / f"CIK{padded_cik}.json"

    if cache_file.exists():
        import json

        with cache_file.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    url = (
        f"https://data.sec.gov/submissions/"
        f"CIK{padded_cik}.json"
    )

    response = sec_get(
        url,
        user_agent,
    )

    data = response.json()

    import json

    with cache_file.open("w", encoding="utf-8") as fh:
        json.dump(
            data,
            fh,
            indent=2,
            sort_keys=True,
        )

    return data


def find_in_scope_filings(
    submissions: dict,
) -> list[dict[str, str]]:
    """Find Q1/Q2 2026 13F filings within the cutoff."""

    recent = submissions["filings"]["recent"]

    filings = []

    allowed_forms = {
        "13F-HR",
        "13F-HR/A",
        "13F-NT",
        "13F-NT/A",
    }

    row_count = len(recent["accessionNumber"])

    for i in range(row_count):
        form = recent["form"][i]
        report_date = recent["reportDate"][i]
        filing_date = recent["filingDate"][i]

        if form not in allowed_forms:
            continue

        if report_date not in REPORT_PERIODS:
            continue

        if filing_date > FILING_DATE_CUTOFF:
            continue

        filings.append(
            {
                "accession_number": recent["accessionNumber"][i],
                "filing_date": filing_date,
                "report_date": report_date,
                "form": form,
                "primary_document": recent["primaryDocument"][i],
            }
        )

    return filings


def run(user_agent: str, output: Path) -> None:
    """Build the dataset."""

    filers = load_filers()

    print(f"Loaded {len(filers)} filers")

    cik_lookup = get_cik_lookup(user_agent)

    print(f"Loaded {len(cik_lookup)} SEC CIK lookup records")

    verified_filers = verify_filers(
        filers,
        cik_lookup,
    )

    write_verified_filers(
        verified_filers,
        output,
    )

    print(
        f"Verified {len(verified_filers)} filers "
        f"and wrote {output / 'filers.csv'}"
    )

    all_filings = []

    for filer in verified_filers:
        submissions = get_submissions(
            filer["cik"],
            user_agent,
        )

        filings = find_in_scope_filings(
            submissions,
        )

        print(
            f"{filer['fund_name']}: "
            f"{len(filings)} in-scope filings"
        )

        for filing in filings:
            filing["fund_name"] = filer["fund_name"]
            filing["cik"] = filer["cik"]

        all_filings.extend(filings)

    print(
        f"\nTotal in-scope filings: {len(all_filings)}"
    )

    if len(all_filings) != 40:
        raise ValueError(
            f"Expected 40 in-scope filings, "
            f"found {len(all_filings)}"
        )

    print(
        f"\nFound all {len(all_filings)} expected "
        f"in-scope filings"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-agent", required=True,
                    help='required by SEC: "FirstName LastName netid@illinois.edu"')
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    if not UA_PATTERN.match(args.user_agent):
        sys.exit(
            "invalid --user-agent.\n"
            "SEC requires a contact address and rejects requests without one.\n"
            '  python main.py --user-agent "Jane Doe jdoe@illinois.edu"'
        )

    args.output.mkdir(parents=True, exist_ok=True)
    run(args.user_agent, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
