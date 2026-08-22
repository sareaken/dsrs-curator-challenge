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
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

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


def download_information_table(
    filing: dict[str, str],
    user_agent: str,
    output: Path,
) -> Path:
    """Download a filing's 13F XML document."""

    cik = filing["cik"]
    accession = filing["accession_number"]
    accession_nodashes = accession.replace("-", "")

    filing_dir_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik}/{accession_nodashes}"
    )

    index_url = f"{filing_dir_url}/index.json"

    index_response = sec_get(
        index_url,
        user_agent,
    )

    index_data = index_response.json()

    xml_candidates = []

    for item in index_data["directory"]["item"]:
        name = item.get("name", "")

        if name.lower().endswith(".xml"):
            xml_candidates.append(name)

    form = filing["form"]

    if form in {"13F-NT", "13F-NT/A"}:
        # Notice filings have no holdings information table.
        # Save the primary XML document instead.
        primary_candidates = [
            name
            for name in xml_candidates
            if name.lower() == "primary_doc.xml"
        ]

        if len(primary_candidates) != 1:
            raise ValueError(
                f"Expected primary_doc.xml for notice filing "
                f"{accession}, found: {xml_candidates}"
            )

        source_filename = primary_candidates[0]

    else:
        # 13F-HR filings normally contain primary_doc.xml
        # plus a separate holdings information-table XML.
        info_candidates = [
            name
            for name in xml_candidates
            if name.lower() != "primary_doc.xml"
        ]

        if len(info_candidates) != 1:
            raise ValueError(
                f"Expected exactly one non-primary XML "
                f"for {accession}, found: {xml_candidates}"
            )

        source_filename = info_candidates[0]

    destination_dir = output / "filings" / cik
    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        destination_dir
        / f"{accession_nodashes}.xml"
    )

    # Reuse a file we've already downloaded.
    if destination.exists():
        return destination

    xml_url = f"{filing_dir_url}/{source_filename}"

    xml_response = sec_get(
        xml_url,
        user_agent,
    )

    destination.write_bytes(
        xml_response.content
    )

    return destination


def get_filing_documents(
    filing: dict[str, str],
    user_agent: str,
) -> tuple[Path, Path | None]:
    """Fetch/cache cover page and information-table XML for one filing."""

    cik = filing["cik"]
    accession = filing["accession_number"]
    accession_nodashes = accession.replace("-", "")

    filing_dir_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik}/{accession_nodashes}"
    )

    filing_cache = (
        CACHE
        / "filings"
        / cik
        / accession_nodashes
    )
    filing_cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------
    # Cache index.json
    # ----------------------------

    index_path = filing_cache / "index.json"

    if index_path.exists():
        import json

        with index_path.open(
            "r",
            encoding="utf-8",
        ) as fh:
            index_data = json.load(fh)

    else:
        index_response = sec_get(
            f"{filing_dir_url}/index.json",
            user_agent,
        )

        index_data = index_response.json()

        import json

        with index_path.open(
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(
                index_data,
                fh,
                indent=2,
                sort_keys=True,
            )

    xml_files = [
        item["name"]
        for item in index_data["directory"]["item"]
        if item.get("name", "").lower().endswith(".xml")
    ]

    # ----------------------------
    # Cover page
    # ----------------------------

    primary_candidates = [
        name
        for name in xml_files
        if name.lower() == "primary_doc.xml"
    ]

    if len(primary_candidates) != 1:
        raise ValueError(
            f"Expected one primary_doc.xml for "
            f"{accession}, found: {xml_files}"
        )

    primary_filename = primary_candidates[0]

    primary_path = filing_cache / "primary_doc.xml"

    if not primary_path.exists():
        response = sec_get(
            f"{filing_dir_url}/{primary_filename}",
            user_agent,
        )

        primary_path.write_bytes(
            response.content
        )

    # ----------------------------
    # Information table
    # ----------------------------

    if filing["form"] in {"13F-NT", "13F-NT/A"}:
        info_path = None

    else:
        info_candidates = [
            name
            for name in xml_files
            if name.lower() != "primary_doc.xml"
        ]

        if len(info_candidates) != 1:
            raise ValueError(
                f"Expected one information-table XML "
                f"for {accession}, found: {xml_files}"
            )

        info_filename = info_candidates[0]

        info_path = filing_cache / "information_table.xml"

        if not info_path.exists():
            response = sec_get(
                f"{filing_dir_url}/{info_filename}",
                user_agent,
            )

            info_path.write_bytes(
                response.content
            )

    return primary_path, info_path


def local_name(element) -> str | None:
    """Return an XML element's local name, skipping comments/PIs."""
    if not isinstance(element.tag, str):
        return None

    return etree.QName(element).localname


def find_first_text(
    element,
    name: str,
) -> str | None:
    """Return text from the first descendant with the given local name."""

    for child in element.iter():
        if local_name(child) == name:
            if child.text is None:
                return None

            value = child.text.strip()

            return value if value else None

    return None


def find_direct_child_text(
    element,
    name: str,
) -> str | None:
    """Return text from a direct child matching a local name."""

    for child in element:
        if local_name(child) == name:
            if child.text is None:
                return None

            value = child.text.strip()

            return value if value else None

    return None


def parse_int(value: str | None) -> int | None:
    """Parse an optional integer, tolerating commas."""

    if value is None:
        return None

    return int(value.replace(",", "").strip())


def parse_filing_cover(
    filing: dict[str, str],
    primary_path: Path,
) -> dict:
    """Parse one filing's cover-page XML."""

    tree = etree.parse(str(primary_path))
    root = tree.getroot()

    report_period_text = find_first_text(
        root,
        "reportCalendarOrQuarter",
    )

    if report_period_text is None:
        raise ValueError(
            f"Missing reportCalendarOrQuarter for "
            f"{filing['accession_number']}"
        )

    report_period = datetime.strptime(
        report_period_text,
        "%m-%d-%Y",
    ).date()

    quarter = ((report_period.month - 1) // 3) + 1

    filing_date = datetime.strptime(
        filing["filing_date"],
        "%Y-%m-%d",
    ).date()

    form_type = filing["form"]
    is_amendment = form_type.endswith("/A")

    filing_manager = find_first_text(
        root,
        "filingManager",
    )

    # filingManager itself contains a child <name>,
    # so retrieve that specifically.
    for element in root.iter():
        if local_name(element) == "filingManager":
            filing_manager = find_direct_child_text(
                element,
                "name",
            )
            break

    if filing_manager is None:
        raise ValueError(
            f"Missing filing manager for "
            f"{filing['accession_number']}"
        )

    return {
        "accession_number": filing["accession_number"],
        "cik": filing["cik"].zfill(10),
        "fund_name": filing["fund_name"],
        "filing_manager": filing_manager,
        "form_type": form_type,
        "report_period": report_period,
        "report_quarter": (
            f"{report_period.year}Q{quarter}"
        ),
        "filing_date": filing_date,
        "is_amendment": is_amendment,

        # We'll populate amendment-specific values if one appears.
        "amendment_no": (
            parse_int(find_first_text(root, "amendmentNo"))
            if is_amendment
            else None
        ),
        "amendment_type": (
            find_first_text(root, "amendmentType")
            if is_amendment
            else None
        ),

        "report_type": find_first_text(
            root,
            "reportType",
        ),
        "form_13f_file_number": find_first_text(
            root,
            "form13FFileNumber",
        ),
        "crd_number": find_first_text(
            root,
            "crdNumber",
        ),
        "sec_file_number": find_first_text(
            root,
            "secFileNumber",
        ),
        "other_included_managers_count": parse_int(
            find_first_text(
                root,
                "otherIncludedManagersCount",
            )
        ),
        "table_entry_total": parse_int(
            find_first_text(
                root,
                "tableEntryTotal",
            )
        ),
        "table_value_total": parse_int(
            find_first_text(
                root,
                "tableValueTotal",
            )
        ),
    }


def parse_holdings(
    filing_row: dict,
    info_path: Path | None,
) -> list[dict]:
    """Parse all positions from one 13F information table."""

    # 13F-NT filings intentionally have no holdings.
    if info_path is None:
        return []

    tree = etree.parse(str(info_path))
    root = tree.getroot()

    holdings = []

    for element in root.iter():
        if local_name(element) != "infoTable":
            continue

        shrs_or_prn = None
        voting_authority = None

        for child in element:
            child_name = local_name(child)

            if child_name == "shrsOrPrnAmt":
                shrs_or_prn = child

            elif child_name == "votingAuthority":
                voting_authority = child

        if shrs_or_prn is None:
            raise ValueError(
                f"Missing shrsOrPrnAmt in "
                f"{filing_row['accession_number']}"
            )

        if voting_authority is None:
            raise ValueError(
                f"Missing votingAuthority in "
                f"{filing_row['accession_number']}"
            )

        cusip = find_direct_child_text(
            element,
            "cusip",
        )

        if cusip is None or len(cusip) != 9:
            raise ValueError(
                f"Invalid CUSIP {cusip!r} in "
                f"{filing_row['accession_number']}"
            )

        holdings.append(
            {
                "accession_number": (
                    filing_row["accession_number"]
                ),
                "cik": filing_row["cik"],
                "report_quarter": (
                    filing_row["report_quarter"]
                ),
                "name_of_issuer": (
                    find_direct_child_text(
                        element,
                        "nameOfIssuer",
                    )
                ),
                "title_of_class": (
                    find_direct_child_text(
                        element,
                        "titleOfClass",
                    )
                ),
                "cusip": cusip,
                "figi": find_direct_child_text(
                    element,
                    "figi",
                ),
                "value": parse_int(
                    find_direct_child_text(
                        element,
                        "value",
                    )
                ),
                "ssh_prnamt": parse_int(
                    find_direct_child_text(
                        shrs_or_prn,
                        "sshPrnamt",
                    )
                ),
                "ssh_prnamt_type": (
                    find_direct_child_text(
                        shrs_or_prn,
                        "sshPrnamtType",
                    )
                ),
                "put_call": find_direct_child_text(
                    element,
                    "putCall",
                ),
                "investment_discretion": (
                    find_direct_child_text(
                        element,
                        "investmentDiscretion",
                    )
                ),
                "other_manager": (
                    find_direct_child_text(
                        element,
                        "otherManager",
                    )
                ),
                "voting_sole": parse_int(
                    find_direct_child_text(
                        voting_authority,
                        "Sole",
                    )
                ),
                "voting_shared": parse_int(
                    find_direct_child_text(
                        voting_authority,
                        "Shared",
                    )
                ),
                "voting_none": parse_int(
                    find_direct_child_text(
                        voting_authority,
                        "None",
                    )
                ),
            }
        )

    return holdings


FILINGS_SCHEMA = pa.schema([
    ("accession_number", pa.string(), False),
    ("cik", pa.string(), False),
    ("fund_name", pa.string(), False),
    ("filing_manager", pa.string(), False),
    ("form_type", pa.string(), False),
    ("report_period", pa.date32(), False),
    ("report_quarter", pa.string(), False),
    ("filing_date", pa.date32(), False),
    ("is_amendment", pa.bool_(), False),
    ("amendment_no", pa.int32(), True),
    ("amendment_type", pa.string(), True),
    ("report_type", pa.string(), False),
    ("form_13f_file_number", pa.string(), True),
    ("crd_number", pa.string(), True),
    ("sec_file_number", pa.string(), True),
    ("other_included_managers_count", pa.int32(), True),
    ("table_entry_total", pa.int64(), True),
    ("table_value_total", pa.int64(), True),
])


HOLDINGS_SCHEMA = pa.schema([
    ("accession_number", pa.string(), False),
    ("cik", pa.string(), False),
    ("report_quarter", pa.string(), False),
    ("name_of_issuer", pa.string(), False),
    ("title_of_class", pa.string(), False),
    ("cusip", pa.string(), False),
    ("figi", pa.string(), True),
    ("value", pa.int64(), False),
    ("ssh_prnamt", pa.int64(), False),
    ("ssh_prnamt_type", pa.string(), False),
    ("put_call", pa.string(), True),
    ("investment_discretion", pa.string(), False),
    ("other_manager", pa.string(), True),
    ("voting_sole", pa.int64(), False),
    ("voting_shared", pa.int64(), False),
    ("voting_none", pa.int64(), False),
])


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

    filing_rows = []
    holding_rows = []
    downloaded_files = []

    for filing in all_filings:
        primary_path, info_path = get_filing_documents(
            filing,
            user_agent,
        )

        # Choose the canonical XML required in output/filings.
        if filing["form"] in {"13F-NT", "13F-NT/A"}:
            source_path = primary_path
        else:
            if info_path is None:
                raise ValueError(
                    f"Missing information table for "
                    f"{filing['accession_number']}"
                )

            source_path = info_path

        cik = filing["cik"]
        accession_nodashes = (
            filing["accession_number"].replace("-", "")
        )

        destination_dir = output / "filings" / cik
        destination_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            destination_dir
            / f"{accession_nodashes}.xml"
        )

        if not destination.exists():
            destination.write_bytes(
                source_path.read_bytes()
            )

        downloaded_files.append(destination)

        filing_row = parse_filing_cover(
            filing,
            primary_path,
        )

        filing_rows.append(filing_row)

        holding_rows.extend(
            parse_holdings(
                filing_row,
                info_path,
            )
        )

    print(
        f"\nDownloaded or reused "
        f"{len(downloaded_files)} filing XML files"
    )

    print(
        f"\nParsed {len(filing_rows)} filings "
        f"and {len(holding_rows)} holdings"
    )

    if len(filing_rows) != 40:
        raise ValueError(
            f"Expected 40 parsed filings, "
            f"found {len(filing_rows)}"
        )


    filings_table = pa.Table.from_pylist(
        filing_rows,
        schema=FILINGS_SCHEMA,
    )

    holdings_table = pa.Table.from_pylist(
        holding_rows,
        schema=HOLDINGS_SCHEMA,
    )

    pq.write_table(
        filings_table,
        output / "filings.parquet",
        compression="snappy",
    )

    pq.write_table(
        holdings_table,
        output / "holdings.parquet",
        compression="snappy",
    )

    print(
        "\nWrote:"
        f"\n  {output / 'filings.parquet'}"
        f"\n  {output / 'holdings.parquet'}"
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
