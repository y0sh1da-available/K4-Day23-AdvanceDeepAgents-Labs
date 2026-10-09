"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


import re

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_URL_REGEX = re.compile(r"https?://\S+")


def _group_numbers(group_str):
    """Expand string like '1, 2' or '1-3' or '1–3' into list of integers."""
    numbers = []
    for part in re.split(r"\s*,\s*", group_str):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    source_ns = set()
    source_urls = set()
    source_by_n = {}

    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"invalid source entry format: {entry!r}")
            continue
        n = entry.get("n")
        url = entry.get("url")
        if not isinstance(n, int):
            problems.append(f"source entry {entry!r} 'n' must be an int")
            continue
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] url must start with http:// or https://: got {url!r}")
            continue

        # Check for placeholder / dummy URLs
        if any(dummy in url.lower() for dummy in ("xyz", "abc", "dummy", "placeholder", "example.com")):
            problems.append(f"source [{n}] contains fake/placeholder url: {url}")

        if url.startswith("https://arxiv.org/abs/"):
            arxiv_id = url.split("/abs/")[-1].strip()
            if not re.match(r"^(\d{4}\.\d{4,5}|[a-zA-Z\-]+/\d{7})$", arxiv_id):
                problems.append(f"source [{n}] has malformed arxiv url: {url}")
        elif url.startswith("https://huggingface.co/papers/"):
            hf_id = url.split("/papers/")[-1].strip()
            if not re.match(r"^\d{4}\.\d{4,5}$", hf_id):
                problems.append(f"source [{n}] has malformed huggingface papers url: {url}")
        if url in source_urls:
            problems.append(f"duplicate url in sources.json: {url}")
        else:
            source_urls.add(url)
        if n in source_ns:
            problems.append(f"duplicate source number in sources.json: [{n}]")
        else:
            source_ns.add(n)
            source_by_n[n] = entry

    # Heading check
    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing heading '## References'")
        return problems

    ref_heading_match = matches[-1]
    body = report_text[:ref_heading_match.start()]
    ref_section = report_text[ref_heading_match.end():]

    # Extract citations in body only, ignoring code spans/blocks and markdown links
    segments = _CODE.split(body)
    cited = set()
    for i, segment in enumerate(segments):
        if i % 2:  # odd indices are code blocks or inline code
            continue
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited.add(num)

    # Check cited numbers against sources
    for num in sorted(cited):
        if num not in source_ns:
            problems.append(f"[{num}] cited but missing from sources.json")

    for n in sorted(source_ns):
        if n not in cited:
            problems.append(f"source [{n}] never cited")

    # Reference lines check
    ref_lines = {}
    for line in ref_section.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        m = re.match(r"^\[(\d+)\]\s*(.*)$", line_str)
        if m:
            n_val = int(m.group(1))
            if n_val in ref_lines:
                problems.append(f"reference list has duplicate entry for [{n_val}]")
            else:
                ref_lines[n_val] = line_str
        elif _URL_REGEX.search(line_str):
            problems.append(f"unformatted reference line in References section: {line_str}")

    for n in sorted(source_ns):
        if n not in ref_lines:
            problems.append(f"missing reference line for source [{n}] in References section")

    for n, line_str in sorted(ref_lines.items()):
        if n not in source_ns:
            problems.append(f"reference line [{n}] is not in sources.json")
            continue

        raw_urls = _URL_REGEX.findall(line_str)
        cleaned_urls = []
        for u in raw_urls:
            # strip trailing punctuation common in references
            u_clean = u.rstrip(").,;")
            cleaned_urls.append(u_clean)

        if len(cleaned_urls) == 0:
            problems.append(f"reference line [{n}] has no URL")
        elif len(cleaned_urls) > 1:
            problems.append(f"reference line [{n}] bundles multiple URLs: {line_str}")
        else:
            expected_url = source_by_n[n].get("url")
            if cleaned_urls[0] != expected_url:
                problems.append(f"reference line [{n}] URL '{cleaned_urls[0]}' does not match sources.json URL '{expected_url}'")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
