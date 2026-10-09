"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time

from agents import (
    FINALIZER_PATH,
    REPORT_PATH,
    SOURCES_PATH,
    VALIDATOR_PATH,
    WORKDIR,
    build_lead_agent,
)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic: str) -> str:
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/.
    """
    s = (topic or "").strip().lower()
    s = re.sub(r"[^\w]+", "-", s).strip("-")
    s = s[:60].rstrip("-")
    return s if s else "topic"


def build_prompt(topic: str) -> str:
    """The user message sent to the lead agent."""
    return (
        f"Conduct a comprehensive, publication-grade academic deep research survey on the topic: '{topic}'.\n\n"
        f"Author: Đặng Hữu Cương\n\n"
        f"Strict Requirements:\n"
        f"1. Plan with `write_todos` and decompose the topic into at least 3 distinct sub-questions.\n"
        f"2. Delegate each sub-question to `researcher` subagents via `task` (must call `task` at least 3 times):\n"
        f"   - Task 1 MUST focus on academic preprints via `arxiv_search` (source family: 'arxiv').\n"
        f"   - Task 2 MUST focus on trending and community papers via `hf_search_papers` or `hf_daily_papers` (source family: 'hf-search' or 'hf-daily').\n"
        f"   - Task 3 MUST focus on web surveys, benchmarks, and project blogs via `web_search` and `web_fetch` (source family: 'web').\n"
        f"3. MULTI-SOURCE CITATION RULE: The final `sources.json` MUST contain AT LEAST 3 DISTINCT source families among ['arxiv', 'hf-daily', 'hf-search', 'web'].\n"
        f"   CRITICAL: In `report.md`, you MUST cite AT LEAST ONE source from 'arxiv', AT LEAST ONE from 'hf-search' (or 'hf-daily'), AND AT LEAST ONE from 'web' in the body text!\n"
        f"   Why? Because `finalize_citations.py` automatically PRUNES uncited sources from `sources.json`. If you don't cite all 3 families in the report body, the uncited family will be deleted, causing the test to fail with 'only 2 source families'.\n"
        f"4. Compile and merge researcher notes into `/tmp/work/research/sources.json` with strictly authentic URLs (no placeholders).\n"
        f"5. Write the report body in academic English at `/tmp/work/report/report.md` following REPORT_TEMPLATE.md "
        f"(with # Title, ## TL;DR, ## Background, 3-6 thematic sections with inline [n] citations from all 3 source families (arxiv, hf, web), and ## Trends and open problems). "
        f"Do NOT write the `## References` section manually.\n"
        f"   CRITICAL CITATION NUMBER RULE: Only cite numbers [n] that exist in `/tmp/work/research/sources.json` (1 to K). Never cite out-of-bounds numbers.\n"
        f"   CHECKLIST BEFORE FINALIZING:\n"
        f"   * Confirm at least one citation points to an arXiv source (source: 'arxiv').\n"
        f"   * Confirm at least one citation points to a Hugging Face source (source: 'hf-search' or 'hf-daily').\n"
        f"   * Confirm at least one citation points to a Web source (source: 'web').\n"
        f"6. Execute `python3 /tmp/work/research/finalize_citations.py` in the sandbox. If it reports '[X] is missing', fix the citations in `report.md` and re-run until it prints 'FINALIZED'.\n"
        f"   CRITICAL: Once finalized, DO NOT edit or alter the `## References` section manually!\n"
        f"7. Execute `python3 /tmp/work/research/check_citations.py` in the sandbox until it returns 'OK: N sources, all citations resolve'.\n"
        f"8. Spot-check 2-3 key claims with `citation-checker` subagent. (Note: spot-checking is for validation only; do NOT edit the References section)."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    Walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_calls = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        t_calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            t_calls = msg.tool_calls
        elif isinstance(msg, dict) and msg.get("tool_calls"):
            t_calls = msg.get("tool_calls")

        for tc in t_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                tool_calls[name] += 1

        usage = None
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            usage = msg.usage_metadata
        elif hasattr(msg, "response_metadata") and isinstance(msg.response_metadata, dict):
            usage = msg.response_metadata.get("token_usage") or msg.response_metadata.get("usage")
        elif isinstance(msg, dict):
            usage = msg.get("usage_metadata") or msg.get("response_metadata", {}).get("token_usage")

        if isinstance(usage, dict):
            in_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            out_tok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            input_tokens += int(in_tok)
            output_tokens += int(out_tok)

    subagent_calls = tool_calls.get("task", 0)

    return {
        "model": str(model_name),
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": int(subagent_calls),
        "tool_calls": dict(tool_calls),
        "tokens": {
            "input": int(input_tokens),
            "output": int(output_tokens),
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    If the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"Report at {REPORT_PATH} is missing or empty")

    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError(f"Sources at {SOURCES_PATH} is missing or empty")

    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources_data, list) or len(sources_data) == 0:
            raise ValueError("sources.json must be a non-empty list")
    except Exception as exc:
        raise RuntimeError(f"Invalid sources.json: {exc}") from exc

    slug = slugify(topic)
    md_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"

    from finalize_citations import finalize
    report_text = report_bytes.decode("utf-8")
    finalized_report, finalized_sources, problems = finalize(report_text, sources_data)
    if not problems and finalized_report:
        report_text = finalized_report
        sources_data = finalized_sources
        report_bytes = report_text.encode("utf-8")
        sources_bytes = json.dumps(sources_data, indent=2, ensure_ascii=False).encode("utf-8")

    source_families = sorted({
        s.get("source") for s in sources_data if isinstance(s, dict) and s.get("source")
    })

    if len(source_families) < 3:
        raise RuntimeError(
            f"sources.json has only {len(source_families)} source families: {source_families}. Need >= 3."
        )

    from check_citations import check
    citation_problems = check(report_text, sources_data)
    if citation_problems:
        raise RuntimeError(
            f"Report failed citation checks: {citation_problems}"
        )

    meta_summary = summarize(messages, elapsed, model_name)
    meta_data = {
        "topic": topic,
        **meta_summary,
        "n_sources": len(sources_data),
        "source_families": source_families,
    }

    sources_path.write_bytes(sources_bytes)
    meta_path.write_text(json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_bytes(report_bytes)

    return md_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    clean_topic = (topic or "").strip()
    if not clean_topic:
        print("Usage: python research.py \"<topic>\"", file=sys.stderr)
        return 2

    model_name = os.getenv("LAB_MODEL", "model")
    model = make_model()

    max_runs = 3
    for run_attempt in range(max_runs):
        start = time.monotonic()
        try:
            with open_sandbox() as backend:
                backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
                upload(
                    backend,
                    {
                        VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                        FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                    },
                )
                agent = build_lead_agent(backend, model)
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": build_prompt(clean_topic)}]},
                    config={"recursion_limit": 1000},
                )
                elapsed = time.monotonic() - start
                messages = result.get("messages", []) if isinstance(result, dict) else []

                saved_path = save_outputs(backend, clean_topic, messages, elapsed, model_name)
                print(f"Report saved successfully to {saved_path}")
                return 0
        except Exception as exc:
            print(f"Run attempt {run_attempt + 1} failed: {exc}", file=sys.stderr)
            if run_attempt < max_runs - 1:
                print("Retrying research run in 5 seconds...", file=sys.stderr)
                time.sleep(5)
            else:
                print(f"FAILED: {exc}", file=sys.stderr)
                return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
