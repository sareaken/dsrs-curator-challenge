#!/usr/bin/env python3
"""Exploratory analysis of the downloaded 13F filings.

This script examines the Chapter 1 XML files and structured outputs to identify
schema mappings, filing variations, and conditions a robust parser must handle.

Findings are printed with concrete counts and example accessions so they can be
reproduced from the committed submission artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
FILINGS_DIR = OUTPUT / "filings"
FILINGS_PARQUET = OUTPUT / "filings.parquet"
HOLDINGS_PARQUET = OUTPUT / "holdings.parquet"


def local_name(element) -> str | None:
    """Return namespace-independent XML local name."""
    if not isinstance(element.tag, str):
        return None

    return etree.QName(element).localname


def xml_files() -> list[Path]:
    """Return all committed filing XML files."""
    return sorted(FILINGS_DIR.glob("*/*.xml"))


def inspect_xml_shapes(paths: list[Path]) -> None:
    """Describe root tags and element-name variation."""

    root_counts = Counter()
    element_presence: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, str] = {}

    for path in paths:
        tree = etree.parse(str(path))
        root = tree.getroot()

        root_name = local_name(root)
        root_counts[root_name] += 1

        accession = path.stem

        for element in root.iter():
            name = local_name(element)

            if name is None:
                continue

            element_presence[name].add(accession)
            examples.setdefault(name, accession)

    print("\n=== XML SHAPES ===")

    print("\nRoot elements:")
    for root_name, count in root_counts.items():
        print(f"  {root_name}: {count}")

    interesting = [
        "infoTable",
        "figi",
        "putCall",
        "otherManager",
        "votingAuthority",
        "None",
    ]

    print("\nSelected element presence:")
    for name in interesting:
        count = len(element_presence.get(name, set()))

        print(
            f"  {name}: present in {count} filing XML files"
        )

        if count:
            print(
                f"    example accession: "
                f"{examples[name]}"
            )


def inspect_filing_types() -> None:
    """Describe filing forms and report types."""

    table = pq.read_table(FILINGS_PARQUET)
    rows = table.to_pylist()

    forms = Counter(row["form_type"] for row in rows)
    report_types = Counter(row["report_type"] for row in rows)

    print("\n=== FILING TYPES ===")

    print("\nForm types:")
    for form, count in sorted(forms.items()):
        print(f"  {form}: {count}")

    print("\nCover-page report types:")
    for report_type, count in sorted(
        report_types.items(),
        key=lambda x: str(x[0]),
    ):
        print(f"  {report_type}: {count}")

    notice_accessions = [
        row["accession_number"]
        for row in rows
        if row["form_type"] in {"13F-NT", "13F-NT/A"}
    ]

    print(
        f"\nNotice filings with no information table: "
        f"{len(notice_accessions)}"
    )

    for accession in notice_accessions[:5]:
        print(f"  {accession}")


def inspect_declared_vs_actual_counts() -> None:
    """Compare declared table-entry totals to parsed row counts."""

    filings = pq.read_table(
        FILINGS_PARQUET
    ).to_pylist()

    holdings = pq.read_table(
        HOLDINGS_PARQUET
    ).to_pylist()

    actual_counts = Counter(
        row["accession_number"]
        for row in holdings
    )

    mismatches = []

    for filing in filings:
        declared = filing["table_entry_total"]

        if declared is None:
            continue

        actual = actual_counts[
            filing["accession_number"]
        ]

        if declared != actual:
            mismatches.append(
                (
                    filing["accession_number"],
                    filing["fund_name"],
                    declared,
                    actual,
                )
            )

    print("\n=== DECLARED VS ACTUAL ENTRY COUNTS ===")

    print(
        f"Holdings filings with count mismatch: "
        f"{len(mismatches)}"
    )

    for accession, fund, declared, actual in mismatches[:10]:
        print(
            f"  {accession} | {fund} | "
            f"declared={declared} actual={actual}"
        )

    if not mismatches:
        print(
            "  No discrepancies found between "
            "table_entry_total and parsed position count."
        )


def inspect_declared_vs_actual_values() -> None:
    """Compare declared table value to sum of holding values."""

    filings = pq.read_table(
        FILINGS_PARQUET
    ).to_pylist()

    holdings = pq.read_table(
        HOLDINGS_PARQUET
    ).to_pylist()

    actual_values = defaultdict(int)

    for row in holdings:
        actual_values[row["accession_number"]] += row["value"]

    mismatches = []

    for filing in filings:
        declared = filing["table_value_total"]

        if declared is None:
            continue

        actual = actual_values[
            filing["accession_number"]
        ]

        if declared != actual:
            mismatches.append(
                (
                    filing["accession_number"],
                    filing["fund_name"],
                    declared,
                    actual,
                )
            )

    print("\n=== DECLARED VS ACTUAL VALUES ===")

    print(
        f"Holdings filings with value mismatch: "
        f"{len(mismatches)}"
    )

    for accession, fund, declared, actual in mismatches[:10]:
        print(
            f"  {accession} | {fund} | "
            f"declared={declared:,} "
            f"actual={actual:,}"
        )

    if not mismatches:
        print(
            "  No discrepancies found between "
            "table_value_total and summed holding value."
        )


def inspect_optional_fields() -> None:
    """Measure optional-field presence and formatting variation."""

    holdings = pq.read_table(
        HOLDINGS_PARQUET
    ).to_pylist()

    total = len(holdings)

    figi_present = sum(
        row["figi"] is not None
        for row in holdings
    )

    put_call_present = sum(
        row["put_call"] is not None
        for row in holdings
    )

    other_manager_present = sum(
        row["other_manager"] is not None
        for row in holdings
    )

    put_call_values = Counter(
        row["put_call"]
        for row in holdings
        if row["put_call"] is not None
    )

    discretion_values = Counter(
        row["investment_discretion"]
        for row in holdings
    )

    amount_types = Counter(
        row["ssh_prnamt_type"]
        for row in holdings
    )

    alpha_cusips = [
        row["cusip"]
        for row in holdings
        if row["cusip"]
        and row["cusip"][0].isalpha()
    ]

    leading_zero_cusips = [
        row["cusip"]
        for row in holdings
        if row["cusip"].startswith("0")
    ]

    print("\n=== OPTIONAL / VARIABLE FIELDS ===")

    print(f"Total positions: {total:,}")

    print(
        f"FIGI present: "
        f"{figi_present:,} / {total:,}"
    )

    print(
        f"put_call present: "
        f"{put_call_present:,} / {total:,}"
    )

    print(
        f"other_manager present: "
        f"{other_manager_present:,} / {total:,}"
    )

    print("\nput_call values:")
    for value, count in sorted(put_call_values.items()):
        print(f"  {value!r}: {count:,}")

    print("\nssh_prnamt_type values:")
    for value, count in sorted(amount_types.items()):
        print(f"  {value!r}: {count:,}")

    print("\ninvestment_discretion values:")
    for value, count in sorted(discretion_values.items()):
        print(f"  {value!r}: {count:,}")

    print(
        f"\nCUSIPs beginning with a letter: "
        f"{len(alpha_cusips):,}"
    )

    for value in sorted(set(alpha_cusips))[:10]:
        print(f"  example: {value}")

    print(
        f"\nCUSIPs beginning with zero: "
        f"{len(leading_zero_cusips):,}"
    )

    for value in sorted(set(leading_zero_cusips))[:10]:
        print(f"  example: {value}")


def inspect_manager_names() -> None:
    """Find roster names that differ from legal filing-manager names."""

    filings = pq.read_table(
        FILINGS_PARQUET
    ).to_pylist()

    differences = []

    for row in filings:
        if row["fund_name"] != row["filing_manager"]:
            differences.append(
                (
                    row["accession_number"],
                    row["fund_name"],
                    row["filing_manager"],
                )
            )

    print("\n=== FUND NAME VS FILING MANAGER ===")

    print(
        f"Filings where names differ exactly: "
        f"{len(differences)} / {len(filings)}"
    )

    for accession, fund, manager in differences[:10]:
        print(
            f"  {accession}\n"
            f"    roster: {fund}\n"
            f"    filing: {manager}"
        )


def print_schema_mapping() -> None:
    """Document where each required field originates."""

    print("\n=== SCHEMA MAPPING ===")

    mappings = {
        "filings.accession_number":
            "submissions API",
        "filings.cik":
            "verified roster CIK, padded to 10 characters",
        "filings.fund_name":
            "supplied filers.csv",
        "filings.filing_manager":
            "cover page: filingManager/name",
        "filings.form_type":
            "submissions API",
        "filings.report_period":
            "cover page: reportCalendarOrQuarter",
        "filings.report_quarter":
            "derived from report_period",
        "filings.filing_date":
            "submissions API filingDate",
        "filings.is_amendment":
            "derived from form_type ending in /A",
        "filings.amendment_no":
            "cover page amendmentNo",
        "filings.amendment_type":
            "cover page amendmentType",
        "filings.report_type":
            "cover page reportType",
        "filings.form_13f_file_number":
            "cover page form13FFileNumber",
        "filings.crd_number":
            "cover page crdNumber",
        "filings.sec_file_number":
            "cover page secFileNumber",
        "filings.other_included_managers_count":
            "cover page otherIncludedManagersCount",
        "filings.table_entry_total":
            "cover page tableEntryTotal (declared)",
        "filings.table_value_total":
            "cover page tableValueTotal (declared)",
        "holdings.accession_number":
            "inherited from filing",
        "holdings.cik":
            "denormalized from filing",
        "holdings.report_quarter":
            "denormalized from filing",
        "holdings.name_of_issuer":
            "infoTable/nameOfIssuer",
        "holdings.title_of_class":
            "infoTable/titleOfClass",
        "holdings.cusip":
            "infoTable/cusip",
        "holdings.figi":
            "infoTable/figi when present",
        "holdings.value":
            "infoTable/value",
        "holdings.ssh_prnamt":
            "infoTable/shrsOrPrnAmt/sshPrnamt",
        "holdings.ssh_prnamt_type":
            "infoTable/shrsOrPrnAmt/sshPrnamtType",
        "holdings.put_call":
            "infoTable/putCall when present",
        "holdings.investment_discretion":
            "infoTable/investmentDiscretion",
        "holdings.other_manager":
            "infoTable/otherManager when present",
        "holdings.voting_sole":
            "infoTable/votingAuthority/Sole",
        "holdings.voting_shared":
            "infoTable/votingAuthority/Shared",
        "holdings.voting_none":
            "infoTable/votingAuthority/None",
    }

    for field, source in mappings.items():
        print(f"  {field}: {source}")


def inspect_namespaces(paths: list[Path]) -> None:
    """Inspect namespace URI and prefix variation across filing XML."""

    namespace_patterns = Counter()
    examples = {}

    for path in paths:
        tree = etree.parse(str(path))
        root = tree.getroot()

        # nsmap maps prefix -> namespace URI.
        # None means the filer used a default namespace.
        pattern = tuple(
            sorted(
                (
                    "<default>" if prefix is None else str(prefix),
                    uri,
                )
                for prefix, uri in root.nsmap.items()
                if uri
            )
        )

        namespace_patterns[pattern] += 1
        examples.setdefault(pattern, path.stem)

    print("\n=== XML NAMESPACE VARIATION ===")

    print(
        f"Distinct root namespace patterns: "
        f"{len(namespace_patterns)}"
    )

    for pattern, count in namespace_patterns.items():
        print(f"\n  {count} filing(s)")
        print(f"    example accession: {examples[pattern]}")

        for prefix, uri in pattern:
            print(f"    prefix={prefix!r} uri={uri!r}")

    print(
        "\nParser implication: element matching should use "
        "namespace/local name rather than assuming a literal prefix."
    )


def inspect_duplicate_cusips() -> None:
    """Find legitimate repeated CUSIPs within individual filings."""

    holdings = pq.read_table(
        HOLDINGS_PARQUET
    ).to_pylist()

    by_filing = defaultdict(list)

    for row in holdings:
        by_filing[row["accession_number"]].append(row)

    filings_with_duplicates = []
    duplicate_groups = 0
    extra_rows = 0
    examples = []

    for accession, rows in by_filing.items():
        cusip_counts = Counter(
            row["cusip"] for row in rows
        )

        duplicates = {
            cusip: count
            for cusip, count in cusip_counts.items()
            if count > 1
        }

        if duplicates:
            filings_with_duplicates.append(accession)
            duplicate_groups += len(duplicates)

            for cusip, count in duplicates.items():
                extra_rows += count - 1

                if len(examples) < 10:
                    matching_rows = [
                        row
                        for row in rows
                        if row["cusip"] == cusip
                    ]

                    examples.append(
                        (
                            accession,
                            cusip,
                            count,
                            matching_rows,
                        )
                    )

    print("\n=== DUPLICATE CUSIPS WITHIN FILINGS ===")

    print(
        f"Filings containing at least one repeated CUSIP: "
        f"{len(filings_with_duplicates)}"
    )

    print(
        f"Repeated accession/CUSIP groups: "
        f"{duplicate_groups:,}"
    )

    print(
        f"Rows that would be lost by keeping only one row "
        f"per accession/CUSIP: {extra_rows:,}"
    )

    for accession, cusip, count, rows in examples[:5]:
        print(
            f"\n  accession={accession} "
            f"cusip={cusip} rows={count}"
        )

        for row in rows[:5]:
            print(
                "    "
                f"issuer={row['name_of_issuer']!r}, "
                f"class={row['title_of_class']!r}, "
                f"put_call={row['put_call']!r}, "
                f"amount={row['ssh_prnamt']}, "
                f"discretion={row['investment_discretion']!r}, "
                f"other_manager={row['other_manager']!r}"
            )

    if filings_with_duplicates:
        print(
            "\nParser implication: CUSIP is not a row key. "
            "Deduplicating or aggregating positions by CUSIP "
            "would destroy legitimate source rows."
        )


def inspect_cusip_lengths() -> None:
    """Validate CUSIP lengths without treating CUSIP as numeric."""

    holdings = pq.read_table(
        HOLDINGS_PARQUET
    ).to_pylist()

    length_counts = Counter(
        len(row["cusip"])
        for row in holdings
        if row["cusip"] is not None
    )

    invalid = [
        (
            row["accession_number"],
            row["cusip"],
        )
        for row in holdings
        if row["cusip"] is None
        or len(row["cusip"]) != 9
    ]

    print("\n=== CUSIP LENGTHS ===")

    for length, count in sorted(length_counts.items()):
        print(
            f"  length {length}: {count:,} positions"
        )

    print(
        f"CUSIPs not exactly 9 characters: "
        f"{len(invalid):,}"
    )

    for accession, cusip in invalid[:10]:
        print(
            f"  {accession}: {cusip!r}"
        )


def inspect_raw_put_call(paths: list[Path]) -> None:
    """Inspect put/call capitalization exactly as represented in XML."""

    values = Counter()
    examples = {}

    for path in paths:
        tree = etree.parse(str(path))

        for element in tree.getroot().iter():
            if local_name(element) != "putCall":
                continue

            if element.text is None:
                continue

            value = element.text.strip()

            values[value] += 1
            examples.setdefault(value, path.stem)

    print("\n=== RAW putCall VALUES ===")

    if not values:
        print("  No putCall elements found.")
        return

    for value, count in sorted(values.items()):
        print(
            f"  {value!r}: {count:,} "
            f"(example {examples[value]})"
        )

    normalized = {
        value.lower()
        for value in values
    }

    print(
        f"Distinct raw spellings: {len(values)}"
    )

    print(
        f"Distinct values ignoring case: "
        f"{len(normalized)}"
    )


def inspect_other_manager_examples() -> None:
    """Show that other_manager contains references rather than manager names."""

    holdings = pq.read_table(
        HOLDINGS_PARQUET
    ).to_pylist()

    values = Counter(
        row["other_manager"]
        for row in holdings
        if row["other_manager"] is not None
    )

    print("\n=== OTHER MANAGER REFERENCES ===")

    print(
        f"Distinct non-null other_manager values: "
        f"{len(values):,}"
    )

    print("\nMost common references:")

    for value, count in values.most_common(10):
        print(
            f"  {value!r}: {count:,} positions"
        )

    example_rows = [
        row
        for row in holdings
        if row["other_manager"] is not None
    ][:5]

    print("\nExample positions:")

    for row in example_rows:
        print(
            f"  {row['accession_number']} | "
            f"{row['name_of_issuer']} | "
            f"other_manager={row['other_manager']!r}"
        )


def inspect_combination_reports() -> None:
    """Show filings that incorporate reporting by other managers."""

    filings = pq.read_table(
        FILINGS_PARQUET
    ).to_pylist()

    combination = [
        row
        for row in filings
        if row["report_type"] == "13F COMBINATION REPORT"
    ]

    print("\n=== COMBINATION REPORTS ===")

    print(
        f"13F COMBINATION REPORT filings: "
        f"{len(combination)}"
    )

    for row in combination:
        print(
            f"  {row['accession_number']} | "
            f"{row['fund_name']} | "
            f"other managers declared="
            f"{row['other_included_managers_count']}"
        )


def print_findings() -> None:
    """Summarize the parser-relevant findings produced by this EDA."""

    print("\n" + "=" * 70)
    print("FINDINGS")
    print("=" * 70)

    print("""
1. FILING AND HOLDINGS GRAIN MUST REMAIN SEPARATE

The 40 in-scope filings contain 39 13F-HR filings and one 13F-NT
notice (accession 0001172661-26-003777). The notice has a cover page
but no information table. A flat holdings-only representation would
therefore make a valid filing disappear entirely. The separate filings
and holdings tables are necessary to preserve notice filings.
""")

    print("""
2. XML NAMESPACE PREFIXES ARE NOT STABLE

Six distinct root namespace patterns occur across the downloaded XML.
The same SEC information-table namespace appears as a default namespace,
under n1, and under ns1; one filing also declares an unrelated Crystal
Reports namespace. A parser that assumes a particular literal prefix
would be brittle. Elements should be matched by namespace URI/local
name rather than prefix.
""")

    print("""
3. CUSIP IS NOT A POSITION KEY

Twenty-four filings contain repeated CUSIPs, covering 26,978 repeated
accession/CUSIP groups. Keeping only one row per accession and CUSIP
would discard 71,152 legitimate source rows.

For example, accession 0000902664-26-003485 reports CUSIP 88579Y101
three times: once as equity, once as a call option, and once as a put
option. Every infoTable element must therefore remain a separate row.
""")

    print("""
4. CUSIP MUST BE STORED AS TEXT

All 134,635 observed CUSIPs are exactly nine characters, but 11,137
begin with a letter and 17,434 begin with zero. Numeric coercion would
fail for valid CINS identifiers and would destroy significant leading
zeros for many domestic identifiers.
""")

    print("""
5. OPTIONAL ELEMENTS CANNOT BE ASSUMED PRESENT

Across 134,635 positions, FIGI is present on 54,487 rows, putCall on
31,032 rows, and otherManager on 85,510 rows. Their absence must be
represented as null rather than treated as malformed input.

In the filings examined, raw putCall values are spelled 'Call' and
'Put'. Although no additional capitalization variants occur in this
sample, the parser should not depend on optional putCall being present.
""")

    print("""
6. sshPrnamt DOES NOT ALWAYS MEAN SHARES

The information tables contain 133,509 SH positions and 1,126 PRN
positions. sshPrnamt therefore cannot safely be interpreted or
aggregated as a share count without consulting sshPrnamtType.
""")

    print("""
7. otherManager IS A REFERENCE, NOT A MANAGER NAME

There are 20 distinct non-null otherManager values, including values
such as '1', '4', '11', and '2,1'. These values point back to manager
definitions on the filing rather than directly storing manager names.
Eight filings are 13F COMBINATION REPORTS, reinforcing that this
relationship is materially present in the dataset.
""")

    print("""
8. ROSTER NAME AND LEGAL FILING MANAGER MUST BOTH BE PRESERVED

Fund name and filing-manager name differ by exact string comparison in
28 of 40 filings. Some differences are capitalization or punctuation,
while others are substantive. For example, the roster's
'Tudor Investment Corp' appears on its filings as
'TUDOR INVESTMENT CORP ET AL', and 'The Baupost Group LLC' appears as
'BAUPOST GROUP LLC/MA'. Replacing one field with the other would lose
information needed either for the researcher roster or the legal filing.
""")

    print("""
9. DECLARED TOTALS ARE USEFUL VALIDATION CHECKS

No discrepancies were observed between cover-page tableEntryTotal and
the number of parsed positions, or between tableValueTotal and the sum
of parsed holding values, for the holdings filings examined. These
fields should nevertheless remain declared source values and can be
used as validation checks rather than recomputed replacements.
""")


def main() -> None:
    paths = xml_files()

    if not paths:
        raise SystemExit(
            "No XML files found under output/filings/"
        )

    print(
        f"Examining {len(paths)} filing XML files"
    )

    print_schema_mapping()
    inspect_xml_shapes(paths)
    inspect_filing_types()
    inspect_declared_vs_actual_counts()
    inspect_declared_vs_actual_values()
    inspect_optional_fields()
    inspect_manager_names()

     # Additional parser-risk investigations
    inspect_namespaces(paths)
    inspect_duplicate_cusips()
    inspect_cusip_lengths()
    inspect_raw_put_call(paths)
    inspect_other_manager_examples()
    inspect_combination_reports()
    print_findings()


if __name__ == "__main__":
    main()