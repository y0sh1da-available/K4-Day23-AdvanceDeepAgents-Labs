"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree

from dotenv import load_dotenv
import httpx
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_LAST_ARXIV_CALL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _redact_key(text: str) -> str:
    """Ensure sensitive API keys never leak in exception messages or agent responses."""
    key = os.getenv("EXA_API_KEY")
    if key and key.strip():
        text = text.replace(key.strip(), "***REDACTED***")
    openrouter_key = os.getenv("LAB_API_KEY")
    if openrouter_key and openrouter_key.strip():
        text = text.replace(openrouter_key.strip(), "***REDACTED***")
    return text


def _parse_retry_after(response: httpx.Response) -> float | None:
    val = response.headers.get("Retry-After")
    if val:
        try:
            return float(val)
        except ValueError:
            return None
    return None


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Respects Retry-After header, uses exponential backoff with random jitter,
    caps at `cap` seconds, raises on final attempt without sleeping,
    and does NOT catch non-RetryableError exceptions.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            if e.retry_after is not None:
                delay = min(float(e.retry_after), cap)
            else:
                exp_delay = base * (2 ** attempt)
                jitter = random.uniform(0.0, 0.5 * exp_delay)
                delay = min(exp_delay + jitter, cap)
            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    try:
        words = [w for w in re.findall(r"[a-zA-Z0-9_-]+", query) if w.lower() not in ("and", "or", "not", "all")]
        if not words:
            return "NO RESULTS"

        search_query = " AND ".join(f"all:{w}" for w in words[:6])
        limit = max(1, min(int(max_results), 30))

        def _do_arxiv():
            global _LAST_ARXIV_CALL
            now = time.monotonic()
            elapsed = now - _LAST_ARXIV_CALL
            if elapsed < 3.0:
                time.sleep(3.0 - elapsed)
            _LAST_ARXIV_CALL = time.monotonic()

            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(
                        ARXIV_URL,
                        params={
                            "search_query": search_query,
                            "sortBy": "submittedDate",
                            "sortOrder": "descending",
                            "max_results": limit,
                        },
                    )
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise RetryableError(
                            f"HTTP {resp.status_code}: {resp.text[:200]}",
                            retry_after=_parse_retry_after(resp),
                        )
                    resp.raise_for_status()
                    return resp.text
            except httpx.TransportError as exc:
                raise RetryableError(f"arXiv transport error: {exc}") from exc

        xml_text = with_retry(_do_arxiv, attempts=6, base=2.0, cap=60.0)

        root = xml.etree.ElementTree.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            id_elem = entry.find("atom:id", ns)
            pub_elem = entry.find("atom:published", ns)
            title_elem = entry.find("atom:title", ns)
            summary_elem = entry.find("atom:summary", ns)

            if id_elem is None or not id_elem.text:
                continue

            raw_id = id_elem.text.strip().split("/abs/")[-1]
            paper_id = re.sub(r"v\d+$", "", raw_id)
            url = f"https://arxiv.org/abs/{paper_id}"

            published = (pub_elem.text.strip() if pub_elem is not None and pub_elem.text else "")[:10]
            title = " ".join((title_elem.text if title_elem is not None and title_elem.text else "").split())
            summary = " ".join((summary_elem.text if summary_elem is not None and summary_elem.text else "").split())[:600]

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        lim = max(1, min(int(limit), 100))
        params = {"limit": lim}
        if date and date.strip():
            params["date"] = date.strip()

        def _do_get():
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(HF_DAILY_URL, params=params)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise RetryableError(
                            f"HTTP {resp.status_code}: {resp.text[:200]}",
                            retry_after=_parse_retry_after(resp),
                        )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TransportError as exc:
                raise RetryableError(f"HF transport error: {exc}") from exc

        items = with_retry(_do_get, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list) or not items:
            return "NO RESULTS"

        kw = keyword.strip().lower()
        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            p = item.get("paper") if isinstance(item.get("paper"), dict) else item
            p_id = p.get("id") or item.get("id")
            if not p_id:
                continue

            title = " ".join(str(p.get("title") or item.get("title") or "").split())
            summary = " ".join(str(p.get("summary") or item.get("summary") or "").split())[:600]
            published = str(p.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = int(p.get("upvotes") or item.get("upvotes") or 0)
            github = str(p.get("githubRepo") or item.get("githubRepo") or "")
            stars = int(p.get("githubStars") or item.get("githubStars") or 0)

            if kw:
                combined = f"{title} {summary}".lower()
                if kw not in combined:
                    continue

            records.append({
                "id": str(p_id),
                "url": f"https://huggingface.co/papers/{p_id}",
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"

        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        lim = max(1, min(int(limit), 50))
        params = {"q": query.strip(), "limit": lim}

        def _do_search():
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(HF_SEARCH_URL, params=params)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise RetryableError(
                            f"HTTP {resp.status_code}: {resp.text[:200]}",
                            retry_after=_parse_retry_after(resp),
                        )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TransportError as exc:
                raise RetryableError(f"HF transport error: {exc}") from exc

        items = with_retry(_do_search, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list) or not items:
            return "NO RESULTS"

        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            p = item.get("paper") if isinstance(item.get("paper"), dict) else item
            p_id = p.get("id") or item.get("id")
            if not p_id:
                continue

            raw_summary = p.get("ai_summary") or p.get("summary") or item.get("ai_summary") or item.get("summary") or ""
            summary = " ".join(str(raw_summary).split())[:600]
            title = " ".join(str(p.get("title") or item.get("title") or "").split())
            published = str(p.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = int(p.get("upvotes") or item.get("upvotes") or 0)
            github = str(p.get("githubRepo") or item.get("githubRepo") or "")
            stars = int(p.get("githubStars") or item.get("githubStars") or 0)

            records.append({
                "id": str(p_id),
                "url": f"https://huggingface.co/papers/{p_id}",
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa(mcp_tool_name: str, arguments: dict) -> dict:
    raw_key = os.getenv("EXA_API_KEY")
    key = raw_key.strip() if raw_key else ""
    url = f"{EXA_URL}?exaApiKey={key}" if key else EXA_URL

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": mcp_tool_name,
            "arguments": arguments,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    def _do_post():
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RetryableError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}",
                        retry_after=_parse_retry_after(resp),
                    )
                resp.raise_for_status()

                data = None
                for line in resp.text.splitlines():
                    if line.startswith("data:"):
                        data = json.loads(line[5:].strip())
                        break
                if data is None:
                    try:
                        data = resp.json()
                    except Exception:
                        pass

                if not data:
                    raise RetryableError("Empty response from Exa")

                if "error" in data:
                    err_obj = data["error"]
                    err_msg = str(err_obj)
                    if "rate" in err_msg.lower() or "429" in err_msg:
                        raise RetryableError(f"Exa rate limit error: {err_msg}", retry_after=20.0)
                    raise RuntimeError(f"Exa RPC error: {err_msg}")

                result = data.get("result", {})
                meta = result.get("_meta", {})
                meta_str = str(meta).lower()
                if meta.get("rate_limited") or "rate limit" in meta_str or "ratelimit" in meta_str:
                    raise RetryableError("Exa rate limit flag in result._meta", retry_after=20.0)

                content_list = result.get("content", [])
                for c in content_list:
                    c_text = (c.get("text") or "").lower()
                    if "rate limit" in c_text and (
                        "exceeded" in c_text or "too many requests" in c_text or "temporarily" in c_text
                    ):
                        raise RetryableError("Exa rate limit message in content", retry_after=20.0)

                return result
        except httpx.TransportError as exc:
            raise RetryableError(f"Exa transport error: {exc}") from exc

    return with_retry(_do_post, attempts=5, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        q = query.strip()
        if not q:
            return "NO RESULTS"
        obj = objective.strip() if objective and objective.strip() else f"Search for {q}"
        num = max(1, min(int(num_results), 10))

        result = _call_exa("web_search_exa", {"query": q, "objective": obj, "numResults": num})
        content_list = result.get("content", [])
        text_parts = [c.get("text", "") for c in content_list if c.get("type") == "text" and c.get("text")]
        combined_text = "\n\n".join(text_parts).strip()

        if not combined_text:
            return "NO RESULTS"
        return combined_text
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        u = url.strip()
        if not u:
            return "NO RESULTS"

        result = _call_exa("web_fetch_exa", {"urls": [u]})
        content_list = result.get("content", [])
        text_parts = [c.get("text", "") for c in content_list if c.get("type") == "text" and c.get("text")]
        combined_text = "\n\n".join(text_parts).strip()

        if not combined_text:
            return "NO RESULTS"
        return combined_text[:12000]
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
