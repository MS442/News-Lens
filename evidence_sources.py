"""Public-source retrieval in Python; retrieved text is data, never instructions.

Exa supplies search results, not verdicts. Our locally trained model performs NLI.
URL extraction pins each connection to a validated public IP, including redirects.
"""
from __future__ import annotations

import hashlib
import html
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

MAX_BYTES = 2_000_000
USER_AGENT = "NewsLens/2.0 (educational article reader)"
_cache: dict = {}
_cache_lock = threading.Lock()
ENRICHMENT_TIMEOUT = 6.0
# Admission slots prevent an executor backlog when an operating-system DNS
# lookup outlives the caller's budget. Busy readers fall back to excerpts.
_enrichment_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="news-source")
_enrichment_slots = threading.BoundedSemaphore(2)


class SourceError(ValueError):
    """A public source cannot safely or reliably be read."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_source_url(url):
    """Ignore protocol/www/tracking aliases, preserve meaningful query parameters."""
    parts = urlsplit(url)
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                   if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid", "mc_cid", "mc_eid"})
    return ((parts.hostname or "").lower().removeprefix("www."), parts.path.rstrip("/") or "/", urlencode(query))


def public_url(value: str, *, resolve: bool = True):
    """Reject credentials, local/reserved addresses, unusual ports and bad URLs."""
    if len(value) > 2048 or re.search(r"[\x00-\x20\\]", value):
        raise SourceError("Enter a complete public http or https article link.")
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").encode("idna").decode("ascii")
        port = parts.port if parts.port is not None else (443 if parts.scheme == "https" else 80)
    except (ValueError, UnicodeError) as exc:
        raise SourceError("This link is not valid.") from exc
    if parts.scheme not in {"http", "https"} or not host or parts.username is not None or parts.password is not None:
        raise SourceError("Only public http or https article links are supported.")
    if port != (443 if parts.scheme == "https" else 80):
        raise SourceError("Only standard web ports are supported.")
    if host.lower() == "localhost" or host.lower().endswith((".localhost", ".local", ".internal")):
        raise SourceError("Private network links are not supported.")
    try:
        literal = ipaddress.ip_address(host)
        if not literal.is_global or literal.is_multicast:
            raise SourceError("Private network links are not supported.")
    except ValueError as exc:
        if isinstance(exc, SourceError):
            raise
    addresses = []
    if resolve:
        try:
            addresses = list(dict.fromkeys(row[4][0] for row in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
        except OSError as exc:
            raise SourceError("The article website could not be reached.") from exc
        if not addresses or any(not ipaddress.ip_address(ip).is_global or ipaddress.ip_address(ip).is_multicast for ip in addresses):
            raise SourceError("This address does not resolve to a public website.")
    clean = urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))
    return clean, host, port, addresses


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, port, address):
        super().__init__(host, port, timeout=5, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        sock = socket.create_connection((self.address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTP(http.client.HTTPConnection):
    def __init__(self, host, port, address):
        super().__init__(host, port, timeout=5)
        self.address = address

    def connect(self):
        self.sock = socket.create_connection((self.address, self.port), self.timeout)


def fetch_public_html(url: str, *, timeout: float = 15.0):
    """Bounded download, no cookies/auth/proxies, revalidate every redirect.

    The elapsed budget is checked around DNS, redirects and reads, and socket
    waits use its remaining time. OS DNS calls cannot be forcibly interrupted;
    this is a practical network budget, not a promise of a hard thread kill.
    """
    started = time.monotonic()
    deadline = started + max(.05, float(timeout))

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise SourceError("The publisher took too long to respond. Paste the text instead.")
        return min(5.0, value)

    def apply_timeout(connection):
        connection.timeout = remaining()
        if connection.sock is not None:
            connection.sock.settimeout(connection.timeout)

    for _ in range(4):
        remaining()
        url, host, port, addresses = public_url(url)
        remaining()
        parts = urlsplit(url)
        connection = (_PinnedHTTPS if parts.scheme == "https" else _PinnedHTTP)(host, port, addresses[0])
        try:
            apply_timeout(connection)
            path = parts.path + ("?" + parts.query if parts.query else "")
            connection.request("GET", path, headers={"Host": parts.netloc, "User-Agent": USER_AGENT,
                                                      "Accept": "text/html,text/plain", "Accept-Encoding": "identity"})
            apply_timeout(connection)
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise SourceError("The article redirected without a destination.")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise SourceError(f"The publisher returned HTTP {response.status}. Paste the article text instead.")
            media = response.getheader("Content-Type", "").lower()
            if not any(t in media for t in ("text/html", "application/xhtml+xml", "text/plain")):
                raise SourceError("Use an HTML article page. PDF and other downloads are not supported here.")
            if response.getheader("Content-Encoding", "identity") not in {"", "identity"}:
                raise SourceError("This publisher requires an unsupported download format. Paste the text instead.")
            if int(response.getheader("Content-Length", "0")) > MAX_BYTES:
                raise SourceError("The page is too large. Paste the article text instead.")
            chunks, size = [], 0
            while True:
                apply_timeout(connection)
                # read1 avoids waiting for an entire 64 KiB buffer when a page
                # trickles data; elapsed time is checked between socket reads.
                reader = getattr(response, "read1", response.read)
                chunk = reader(65536)
                remaining()
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise SourceError("The page is too large. Paste the article text instead.")
                chunks.append(chunk)
            return url, b"".join(chunks)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            if isinstance(exc, SourceError):
                raise
            raise SourceError("The publisher could not be read. Paste the article text instead.") from exc
        finally:
            connection.close()
    raise SourceError("The link redirects too many times. Paste the article text instead.")


def extract_article(url: str, *, timeout: float | None = None):
    final_url, raw = fetch_public_html(url) if timeout is None else fetch_public_html(url, timeout=timeout)
    soup = BeautifulSoup(raw, "html.parser")
    title_node = soup.find("h1") or soup.find("title")
    title = title_node.get_text(" ", strip=True)[:240] if title_node else urlsplit(final_url).hostname
    published = soup.find("meta", attrs={"property": "article:published_time"})
    published_at = published.get("content") if published else None
    article_body = None
    for node in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            objects = json.loads(node.string or node.get_text())
            queue = objects if isinstance(objects, list) else [objects]
            for obj in queue:
                if not isinstance(obj, dict):
                    continue
                queue.extend(obj.get("@graph", []) if isinstance(obj.get("@graph"), list) else [])
                if isinstance(obj.get("articleBody"), str) and len(obj["articleBody"]) > 100:
                    article_body = obj["articleBody"]
                    published_at = obj.get("datePublished", published_at)
                    break
        except (TypeError, ValueError):
            continue
    for node in soup.select("script,style,nav,footer,aside,form,button,noscript,svg,iframe"):
        node.decompose()
    body = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    paragraphs = [n.get_text(" ", strip=True) for n in body.find_all(["p", "h2", "h3", "li"])]
    paragraphs = [p for p in paragraphs if len(p.split()) >= 5]
    text = article_body or "\n\n".join(dict.fromkeys(paragraphs)) or body.get_text(" ", strip=True)
    text = re.sub(r"[ \t]+", " ", text).strip()[:29500]
    if len(text.split()) < 20:
        raise SourceError("There was not enough readable article text. Paste the article instead.")
    return {"title": title, "text": f"{title}\n\n{text}"[:30000], "source_url": final_url,
            "domain": urlsplit(final_url).hostname, "published_at": published_at,
            "word_count": len(re.findall(r"\b[\w'-]+\b", text))}


def _plain_title(value):
    """Search providers can include entity-encoded inline HTML in a title."""
    decoded = html.unescape(str(value or ""))
    if "<" in decoded and ">" in decoded:
        soup = BeautifulSoup(decoded, "html.parser")
        for node in soup.select("script,style"):
            node.decompose()
        decoded = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", decoded).strip()[:250]


def enrich_sources(sources, excluded_url=None, limit=2):
    """Read up to two web pages concurrently, retaining search excerpts on failure.

    The caller waits at most ENRICHMENT_TIMEOUT seconds for the workers. Running
    DNS/socket work cannot be killed safely, so late results are ignored and the
    shared pool stays capped at two admitted jobs with no growing work queue.
    Busy workers cause immediate excerpt fallback. Supplied sources are never fetched.
    Only `text` contains source content; metadata can be returned to the UI
    without accidentally exposing a second full-text/search-text field.
    """
    deadline = time.monotonic() + ENRICHMENT_TIMEOUT
    excluded = canonical_source_url(excluded_url) if excluded_url else None
    records, warnings = [], []
    for source in sources:
        row = dict(source)
        if row.get("url") and excluded and canonical_source_url(row["url"]) == excluded:
            warnings.append("The article itself was excluded as evidence; it cannot corroborate itself.")
            continue
        if row.get("kind") == "web":
            row["title"] = _plain_title(row.get("title"))
            row["enrichment_status"] = "search_excerpt"
            row["original_requested_url"] = row.get("original_requested_url") or row.get("url")
        records.append(row)

    candidates = [index for index, row in enumerate(records) if row.get("kind") == "web" and row.get("url")]
    candidates = candidates[:max(0, min(2, int(limit)))]
    if not candidates:
        return records, warnings

    def read_source(index):
        source = records[index]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SourceError("The full-page reading time budget expired before the reader started.")
        extracted = extract_article(source["url"], timeout=remaining)
        if not isinstance(extracted.get("text"), str) or len(extracted["text"].split()) < 20:
            raise SourceError("The page did not provide enough readable article text.")
        final_url = extracted["source_url"]
        if excluded and canonical_source_url(final_url) == excluded:
            return None
        return {**source, "text": extracted["text"], "url": final_url,
                "title": _plain_title(extracted.get("title") or source.get("title")),
                "domain": urlsplit(final_url).hostname,
                "published_at": extracted.get("published_at") or source.get("published_at"),
                "enrichment_status": "full_text", "page_retrieved_at": utc_now()}

    futures = {}
    slots = _enrichment_slots
    for index in candidates:
        if not slots.acquire(blocking=False):
            warnings.append(f"Full-page readers are busy for {records[index].get('domain') or 'a source'}; its search excerpt was retained.")
            continue
        try:
            future = _enrichment_pool.submit(read_source, index)
        except BaseException:
            slots.release()
            raise
        # A running reader keeps its slot after the caller times out. Completion,
        # exceptions and cancellation all release it exactly once.
        future.add_done_callback(lambda _future, slots=slots: slots.release())
        futures[future] = index
    if not futures:
        return records, warnings
    done, pending = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
    removed = set()
    for future in done:
        index = futures[future]
        try:
            enriched = future.result()
        except (SourceError, OSError, ValueError, KeyError, TypeError):
            warnings.append(f"Could not read the full page from {records[index].get('domain') or 'a source'}; its search excerpt was retained.")
            continue
        if enriched is None:
            removed.add(index)
            warnings.append("A source redirected to the submitted article and was excluded as evidence.")
        else:
            records[index] = enriched
    for future in pending:
        future.cancel()
        warnings.append(f"Full-page reading reached its time budget for {records[futures[future]].get('domain') or 'a source'}; its search excerpt was retained.")
    return [row for index, row in enumerate(records) if index not in removed], warnings


def _rpc_result(response: httpx.Response):
    rate_limit_message = ("Web search has reached Exa's rate limit. Try again later or add source evidence text. "
                          "The app owner can configure an Exa API key with available quota.")
    if response.status_code == 429:
        raise SourceError(rate_limit_message)
    if response.status_code != 200:
        raise SourceError("Web search is unavailable. Add a source passage or try again later.")
    if len(response.content) > MAX_BYTES:
        raise SourceError("The search response was too large.")
    try:
        if "text/event-stream" in response.headers.get("content-type", ""):
            messages = [json.loads(line[5:].strip()) for line in response.text.splitlines() if line.startswith("data:")]
            data = next(item for item in messages if isinstance(item, dict) and ("result" in item or "error" in item))
        else:
            data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("result"), dict):
            raise SourceError("Web search returned an unreadable response. Try again later.")
        metadata = data["result"].get("_meta")
        # Exa's free MCP quota response can be HTTP 200 without isError. Its
        # provider metadata, rather than article text, identifies the failure.
        if isinstance(metadata, dict) and metadata.get("ai.exa/rateLimited") is True:
            raise SourceError(rate_limit_message)
        if "error" in data or data["result"].get("isError"):
            raise SourceError("Web search could not finish. Add evidence text or try later.")
        return data["result"]
    except (ValueError, KeyError, StopIteration) as exc:
        if isinstance(exc, SourceError):
            raise
        raise SourceError("Web search returned an unreadable response. Try again later.") from exc


def search_sources(query: str, count: int = 5):
    """Real Exa search via its documented public, rate-limited MCP endpoint.

    An optional EXA_API_KEY uses the account's own allocation. We send only the
    displayed short query; no LLM service decides the classification.
    """
    key = hashlib.sha256(query.encode()).hexdigest()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and cached[0] > time.monotonic():
            return cached[1]
    headers = {"Accept": "application/json, text/event-stream", "User-Agent": USER_AGENT}
    api_key = os.environ.get("EXA_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "web_search_exa", "arguments": {"query": query[:450], "numResults": count,
        "objective": "Find primary sources and independent reporting that support or contradict the factual details. Return relevant passages and original URLs. Avoid social posts, opinion, and copies without evidence."}}}
    try:
        with httpx.Client(timeout=22, trust_env=bool(os.environ.get("NEWSLENS_TRUST_ENV"))) as client:
            result = _rpc_result(client.post("https://mcp.exa.ai/mcp", headers=headers, json=payload))
    except httpx.HTTPError as exc:
        raise SourceError("Web search timed out or is unavailable. Add evidence text or try later.") from exc
    content = "\n".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text")
    sources = []
    for block in re.split(r"(?m)(?=^Title: )", content):
        title = re.search(r"(?m)^Title: (.+)$", block)
        url = re.search(r"(?m)^URL: (https?://\S+)", block)
        passage = re.split(r"(?m)^(?:Highlights|Text|Content):\s*\n?", block, maxsplit=1)
        if not title or not url or len(passage) < 2:
            continue
        try:
            clean, host, _, _ = public_url(url.group(1), resolve=False)
        except SourceError:
            continue
        published = re.search(r"(?m)^Published: (.+)$", block)
        body = passage[1].strip().removesuffix("---").strip()[:14000]
        if len(body.split()) < 12:
            continue
        sources.append({"title": _plain_title(title.group(1)), "url": clean, "domain": host,
                        "published_at": published.group(1) if published and published.group(1) != "N/A" else None,
                        "retrieved_at": utc_now(), "kind": "web", "text": body})
    if not sources:
        raise SourceError("Search returned no usable source passages. Add evidence from a source you trust.")
    with _cache_lock:
        if len(_cache) >= 64:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (time.monotonic() + 900, sources)
    return sources
