"""Explainable Python orchestration: split -> retrieve -> rank -> compare.

Sentence selection and TF-IDF retrieval are NLP rules, not deep learning. Only
EvidenceClassifier's Transformer performs learned evidence/claim inference.
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.parse import urlsplit

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from evidence_sources import SourceError, canonical_source_url, enrich_sources, extract_article, public_url, search_sources, utc_now

MAX_CLAIMS = 8
MAX_SOURCES = 6
MAX_SOURCE_CHARS = 30000
PASSAGES_PER_CLAIM = 6
SEARCH_TIMEOUT = 22.0
_search_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="news-search")
_search_slots = threading.BoundedSemaphore(6)
LABELS = {"supported": "Supported by this evidence", "contradicted": "Contradicted by this evidence",
          "conflicting": "Conflicting source evidence", "insufficient_evidence": "Insufficient evidence"}


def ranking_text(text):
    """Equivalent number/unit spellings only for retrieval, never rewrite NLI input."""
    units = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
    tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
    text = text.lower()
    text = re.sub(r"\b(" + "|".join(tens) + r")[ -](one|two|three|four|five|six|seven|eight|nine)\b",
                  lambda m: str(tens[m[1]] + units[m[2]]), text)
    mapping = {**units, **tens}
    text = re.sub(r"\b(" + "|".join(mapping) + r")\b", lambda m: str(mapping[m[0]]), text)
    return re.sub(r"\bmet(?:re|er)s?\b", "meter", text)


def sentences(text):
    # Keep decimals and common abbreviations intact. This is intentionally a
    # small, inspectable segmenter, not an undocumented generative model.
    protected = re.sub(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|U\.S|U\.K)\.",
                       lambda match: match.group().replace(".", "\u2024"), text)
    protected = re.sub(r"\b[A-Z]\.", lambda match: match.group().replace(".", "\u2024"), protected)
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])|\n\s*\n", protected)
    return [re.sub(r"\s+", " ", chunk).replace("\u2024", ".").strip() for chunk in chunks if chunk.strip()]


def select_claims(text):
    lines = text.strip().splitlines()
    title = lines[0].strip()
    has_title = len(lines) > 1 and len(title.split()) <= 20 and not title.endswith((".", "!", "?"))
    body = "\n".join(lines[1:]).strip() if has_title else text
    all_sentences = sentences(body)
    eligible = [part for part in all_sentences if len(part.split()) >= 5 and len(part) >= 25 and not part.endswith("?")]
    if not eligible:
        # A yes/no question ("Did X say Y?") is checked as the statement it asks about.
        eligible = [part.rstrip("?").strip() for part in all_sentences if len(part.split()) >= 5]
    selected = eligible[:MAX_CLAIMS]
    return {"title": title if has_title else "", "claims": selected,
            "total_sentences": len(all_sentences), "omitted": len(all_sentences) - len(selected)}


PRONOUN_START = re.compile(r"^(It|He|She|They|This|These|Its|His|Her|Their)\b(?:'s)?\s+", re.I)
SUBJECT_VERB = re.compile(r"\b(?:is|was|are|were|has|have|had|will|would|can|could|said|says|became|becomes|"
                          r"announced|launched|won|lost|died|opened|located|stands|remains)\b")


def subject_of(sentence):
    """Leading noun phrase of a sentence, e.g. 'The Eiffel Tower is ...'."""
    match = SUBJECT_VERB.search(sentence)
    if not match:
        return None
    subject = sentence[:match.start()].strip(" ,")
    words = subject.split()
    if not 1 <= len(words) <= 7 or not subject[0].isupper() or PRONOUN_START.match(subject + " "):
        return None
    return subject


def resolve_claim(claim, previous):
    """Replace a leading pronoun with the earlier subject so the claim stands alone.

    'It was completed in 1889' cannot be retrieved or verified on its own; the
    displayed text is unchanged, only the retrieval/NLI hypothesis is resolved.
    """
    match = PRONOUN_START.match(claim)
    if not match:
        return claim
    for sentence in reversed(previous):
        subject = subject_of(sentence)
        if subject:
            word = match.group(1).lower()
            if word in {"its", "his", "her", "their"}:
                return f"{subject}'s {claim[match.end():]}"
            if word in {"this", "these"}:
                return claim  # demonstratives usually refer to an event, not the subject
            return f"{subject} {claim[match.end():]}"
    return claim


SPEECH_VERBS = r"(?:told|asked|warned|informed|threatened|reminded|assured|promised)"
# "X told PM Modi that ..." / "X told reporters that ..." -> "X said that ..."
REPORTED_SPEECH = re.compile(SPEECH_VERBS + r"\s+(?:[\w.'-]+\s+){1,4}?that\s+", re.I)
# "X told PM Modi 'tera baap bhi jayega'" (no "that")
REPORTED_SPEECH_NAMED = re.compile("(?i:" + SPEECH_VERBS + r")\s+(?:(?:PM|Prime Minister|President|Mr\.?|Ms\.?|Dr\.?|Shri|the)\s+)*"
                                   r"(?:[A-Z][\w.'-]*\s+){1,3}")
# Second-person / possessive words that a paraphrased quote often swaps
# ("tera baap" vs "Modi ka baap", "your father" vs "his father").
SWAPPABLE = r"(?:tera|teri|tere|tum|tumhara|tumhari|aap|aapka|aapki|apna|apni|your|you|his|her|their|my)"


def relaxed_claims(claim):
    """Core-claim variants used only when the full claim is unresolved.

    A news claim often bundles the core event with details (who it was said
    to, the exact wording). The NLI model calls the whole claim a
    contradiction when one detail differs, even if the core event is well
    reported. These variants drop those details so the core can be checked,
    and the result is labelled as partial, never as full support.
    """
    variants = []
    speech = REPORTED_SPEECH.sub("said that ", claim, count=1)
    if speech == claim:
        speech = REPORTED_SPEECH_NAMED.sub("said that ", claim, count=1)
    if speech != claim:
        variants.append(speech)
    for base in [speech, claim]:
        quote = re.sub(r"(\bthat\s+|[\"'\u201c\u2018]\s*)" + SWAPPABLE + r"\s+", r"\1", base, count=1, flags=re.I)
        if quote != base:
            variants.append(quote)
    return [v for v in dict.fromkeys(variants) if v != claim and len(v.split()) >= 4]


def query_for(article):
    # Limit external disclosure to a short headline/first-three-sentence query.
    prefix = article["title"] + ". " if article["title"] else ""
    first = " ".join(article["claims"][:3])
    return ("Reports and primary sources about: " + prefix + first)[:450]


def queries_for(article):
    """One opening query and at most two focused groups for later statements."""
    queries = [query_for(article)]
    later = article["claims"][3:]
    # Later claims were absent from the opening query. Cover two bounded groups
    # rather than issuing one network request per statement.
    if later:
        width = (len(later) + 1) // 2
        for start in range(0, len(later), width):
            group = later[start:start + width]
            query = "Evidence and original reporting for: " + " ".join(group)
            if article["title"]:
                query += " Topic: " + article["title"]
            queries.append(query[:450])
    return list(dict.fromkeys(queries))[:3]


def retrieve_queries(queries):
    """Bound caller wait and worker capacity; late work never grows a queue."""
    deadline = time.monotonic() + SEARCH_TIMEOUT
    batches, errors, futures = [[] for _ in queries], {}, {}
    for index, query in enumerate(queries):
        if time.monotonic() >= deadline:
            errors[index] = "Search time limit reached; this query was skipped."
            continue
        # A timed-out worker retains its slot until it actually finishes. A
        # later request can therefore fall back immediately instead of queuing.
        slots = _search_slots
        if not slots.acquire(blocking=False):
            errors[index] = "Search is busy; this query was skipped. Available evidence is still compared."
            continue
        try:
            future = _search_pool.submit(search_sources, query)
        except RuntimeError:
            slots.release()
            errors[index] = "Search is unavailable; this query was skipped."
            continue
        future.add_done_callback(lambda _future, held=slots: held.release())
        futures[future] = index
    _, pending = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
    for future, index in futures.items():
        if future in pending:
            # Running I/O cannot be cancelled by a Future, but neither its
            # completion nor executor shutdown is awaited by this request.
            future.cancel()
            errors[index] = "Search time limit reached; this query's late result was skipped."
            continue
        try:
            batches[index] = future.result()
        except SourceError as error:
            errors[index] = str(error)
        except Exception:
            errors[index] = "A search query failed; available evidence is still compared."
    return batches, [errors[index] for index in sorted(errors)]


def deduplicate_sources(sources):
    """Canonical URL duplicates share one fetch; prefer full text after redirects."""
    unique = {}
    for source in sources:
        key = canonical_source_url(source["url"]) if source["url"] else source["text"]
        previous = unique.get(key)
        if previous is None or (source.get("enrichment_status") == "full_text"
                                and previous.get("enrichment_status") != "full_text"):
            unique[key] = source
    return list(unique.values())


def passage_key(text):
    """Ignore typography/spacing only; retain names, numbers and negation."""
    quotes = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
    return " ".join(text.translate(quotes).casefold().split())


def bounded_chunks(text, words=155):
    """Overlapping slices retain the end of a long sentence, not just its start."""
    spans = list(re.finditer(r"\S+", text))
    if len(spans) <= words:
        return [text]
    result = []
    for start in range(0, len(spans), words - 35):
        end = min(start + words, len(spans))
        result.append(text[spans[start].start():spans[end - 1].end()])
        if end == len(spans):
            break
    return result


def is_statement(text):
    """Skip headings, captions, bylines and nav fragments: they are not evidence
    and NLI models produce confident nonsense on them."""
    words = text.split()
    if len(words) < 6 or text.rstrip().endswith("?"):
        return False
    lowercase = sum(word[:1].islower() for word in words)
    return lowercase >= max(2, len(words) // 4)


def passage_windows(source):
    text = re.sub(r"(?m)^#+\s*", "", source["text"][:MAX_SOURCE_CHARS])
    # Ellipses in search highlights mark gaps. Never silently join those gaps.
    blocks = re.split(r"\n\s*(?:\.\.\.|---)\s*\n", text)
    windows = []
    for block in blocks:
        whole = re.sub(r"\s+", " ", block).strip()
        if is_statement(whole) and len(whole.split()) <= 155:
            windows.append(whole)
        parts = [part for part in sentences(block) if is_statement(part) and "| ---" not in part]
        quoted = [bool(re.search(r'["\u201c\u201d]|(?<!\w)[\u2018\u2019\']', part)) for part in parts]
        for index, part in enumerate(parts):
            # A quote's neighbours often identify who said it, to whom and why.
            # Preserve that source context, without inventing a resolved claim.
            context = " ".join(parts[max(0, index - 1):index + 2])
            if quoted[index] and len(context.split()) <= 155:
                windows.append(context)
            else:
                windows.extend(bounded_chunks(part))
            if (index + 1 < len(parts) and not (quoted[index] or quoted[index + 1])
                    and len((part + " " + parts[index + 1]).split()) <= 155):
                windows.append(part + " " + parts[index + 1])
    # Input text itself is bounded. Rank all of it: clipping the first 70
    # windows before ranking previously made later evidence unreachable.
    unique = {}
    for window in windows:
        unique.setdefault(passage_key(window), window)
    return list(unique.values())


def rank_passages(claim, sources, limit=PASSAGES_PER_CLAIM):
    candidates = []
    for source in sources:
        for passage in passage_windows(source):
            candidates.append({"source_id": source["id"], "passage": passage})
    if not candidates:
        return []
    try:
        stop_words = sorted(word for word in ENGLISH_STOP_WORDS if ranking_text(word) == word)
        vectorizer = TfidfVectorizer(preprocessor=ranking_text, stop_words=stop_words, ngram_range=(1, 2), sublinear_tf=True, max_features=12000,
                                     token_pattern=r"(?u)\b\w+\b")
        vectors = vectorizer.fit_transform([claim] + [item["passage"] for item in candidates])
        relevance = (vectors[1:] @ vectors[0].T).toarray().ravel()
    except ValueError:
        return []
    order = np.argsort(-relevance)
    ranked, seen = [], set()
    for index in order:
        if relevance[index] < .08:
            break
        item = candidates[int(index)]
        normalized = passage_key(item["passage"])
        # Only exact duplicates are safe to suppress. Small edits to negation,
        # units, people or numbers can change the fact being asserted.
        if normalized in seen:
            continue
        seen.add(normalized)
        ranked.append({**item, "relevance": round(float(relevance[index]), 4)})
    # First offer each distinct source one slot, then fill spare slots. This
    # prevents three overlapping snippets from hiding another relevant source.
    selected, source_ids = [], set()
    for item in ranked:
        if item["source_id"] in source_ids:
            continue
        selected.append(item)
        source_ids.add(item["source_id"])
        if len(selected) == limit:
            break
    if len(selected) < limit:
        selected.extend(item for item in ranked if item not in selected)
    return selected[:limit]


GENERIC_CAPS = set("""a an the this that these those it he she they his her their its in on at of for and but or
president former prime minister mr mrs ms dr pm chief senator governor minister leader party
monday tuesday wednesday thursday friday saturday sunday january february march april may june july
august september october november december""".split())


def named_entities(text):
    """Capitalised words (people, places, organisations), as 5-letter stems so
    'Africa'/'African' or 'Kharge'/'Kharge's' match."""
    words = re.findall(r"\b[A-Z][\w'-]+", text)
    return {w.lower().replace("'s", "")[:5] for w in words if w.lower() not in GENERIC_CAPS}


def gate_contradictions(evidence, claim):
    """A passage can only contradict a claim about the same named things.

    Without this, "The former president rallied in Cleveland" (about Obama)
    was read as contradicting "Trump held a rally in Port Huron".
    """
    entities = named_entities(claim)
    if not entities:
        return evidence
    gated = []
    for row in evidence:
        if row["relation"] == "contradiction" and not (entities & named_entities(row["passage"])):
            row = {**row, "relation": "review", "decision": "review",
                   "decision_reason": "Possible contradiction ignored: the passage does not mention the people or places in this statement."}
        gated.append(row)
    return gated


def decide_claim(evidence, have_sources):
    """Aggregate passage-level NLI results into one claim verdict.

    Previously a claim was only resolved when a single passage cleared a 95%
    threshold, and any one stray contradiction made the claim "conflicting".
    Now: strong evidence wins; otherwise moderate evidence from the best passage
    on each side is weighed, and the side with the clear lead decides.
    """
    def best(relation, decisions):
        rows = [row for row in evidence if row["relation"] == relation and row["decision"] in decisions]
        return max((row["scores"][relation] for row in rows), default=0.0), len({row["source_id"] for row in rows})

    strong_support, _ = best("entailment", {"decisive"})
    strong_contra, _ = best("contradiction", {"decisive"})
    support, support_sources = best("entailment", {"decisive", "likely"})
    contra, contra_sources = best("contradiction", {"decisive", "likely"})

    if strong_support and strong_contra:
        return ("conflicting", "strong",
                "Different sources strongly support and contradict this statement. Check dates and context.")
    if strong_support or strong_contra:
        verdict = "supported" if strong_support else "contradicted"
        other = contra if strong_support else support
        note = " A weaker passage points the other way; open the sources." if other else ""
        word = "supports" if strong_support else "contradicts"
        return verdict, "strong", f"A source passage {word} this statement with high model confidence.{note}"
    if support and contra:
        # Both sides only moderate: require a clear lead before choosing.
        if support - contra >= 0.15 or (support_sources > contra_sources and support >= contra):
            return "supported", "moderate", "Most of the compared evidence supports this statement; one passage disagrees."
        if contra - support >= 0.15 or contra_sources > support_sources:
            return "contradicted", "moderate", "Most of the compared evidence contradicts this statement; one passage agrees."
        return "conflicting", "moderate", "Compared passages point in different directions. Read the sources."
    if support:
        n = f" ({support_sources} sources)" if support_sources > 1 else ""
        return "supported", "moderate", f"Source evidence supports this statement with moderate model confidence{n}."
    if contra:
        n = f" ({contra_sources} sources)" if contra_sources > 1 else ""
        return "contradicted", "moderate", f"Source evidence contradicts this statement with moderate model confidence{n}."
    reason = ("No source passages are available to compare." if not have_sources else
              "Sources were found, but no relevant passage was selected for this statement." if not evidence else
              "Relevant passages were found, but none clearly supports or contradicts the statement. This does not mean the claim is false.")
    return "insufficient_evidence", None, reason


def analyse_article(model, *, text, source_url=None, evidence_text=None, evidence_url=None, search_web=True):
    started = time.perf_counter()
    timings = {"retrieval_ms": 0, "enrichment_ms": 0, "ranking_ms": 0, "inference_ms": 0}
    article = select_claims(text)
    if not article["claims"]:
        raise ValueError("Paste at least one complete factual sentence in English.")
    sources, warnings = [], []
    retrieval_started = time.perf_counter()
    # A supplied source is labelled as such, not silently promoted to trusted.
    if evidence_text:
        clean_url = public_url(evidence_url, resolve=False)[0] if evidence_url else None
        sources.append({"title": "Source passage you supplied", "url": clean_url,
                        "domain": urlsplit(clean_url).hostname if clean_url else "Supplied text",
                        "published_at": None, "retrieved_at": utc_now(), "kind": "supplied", "text": evidence_text})
    elif evidence_url:
        try:
            extracted = extract_article(evidence_url)
            sources.append({"title": extracted["title"], "url": extracted["source_url"], "domain": extracted["domain"],
                            "published_at": extracted["published_at"], "retrieved_at": utc_now(), "kind": "supplied", "text": extracted["text"]})
        except SourceError as error:
            warnings.append(str(error))
    search = {"status": "disabled", "queries": [], "provider": "Exa", "message": "Web search is off; only supplied evidence is compared."}
    if search_web:
        queries = queries_for(article)
        search["queries"] = queries
        batches, errors = retrieve_queries(queries)
        # Interleave query results so the opening query does not consume every
        # source slot before a later statement's targeted results are considered.
        for index in range(max((len(batch) for batch in batches), default=0)):
            sources.extend(batch[index] for batch in batches if index < len(batch))
        found_count = sum(len(batch) for batch in batches)
        status = "partial" if errors and found_count else "unavailable" if errors else "ok"
        message = ("Some searches failed; the available source passages are still compared." if status == "partial" else
                   errors[0] if errors else "Search completed but returned no usable source passages." if not found_count else
                   "Retrieved related sources; passages are checked separately for each statement.")
        search.update(status=status, message=message, queries_succeeded=len(queries) - len(errors), queries_failed=len(errors))
        warnings.extend(errors)
    timings["retrieval_ms"] = round((time.perf_counter() - retrieval_started) * 1000)

    # Exclude the input article before spending a fetch on it. Enrichment also
    # checks redirects; final deduplication below rechecks its returned URLs.
    canonical = canonical_source_url(source_url) if source_url else None
    candidates = []
    for source in sources:
        if canonical and source["url"] and canonical_source_url(source["url"]) == canonical:
            warnings.append("The article itself was excluded as evidence; it cannot corroborate itself.")
        else:
            candidates.append(source)
    # Queries commonly return the same canonical article. Deduplicate before
    # enrichment so two aliases do not consume both full-article fetch slots.
    candidates = deduplicate_sources(candidates)
    if search_web and candidates:
        enrichment_started = time.perf_counter()
        candidates, enrichment_warnings = enrich_sources(candidates, excluded_url=source_url, limit=2)
        timings["enrichment_ms"] = round((time.perf_counter() - enrichment_started) * 1000)
        warnings.extend(enrichment_warnings)
    independent = []
    for source in candidates:
        if source_url and source["url"] and canonical_source_url(source["url"]) == canonical_source_url(source_url):
            warnings.append("The article itself was excluded as evidence; it cannot corroborate itself.")
            continue
        independent.append(source)
    sources = [{**source, "id": f"S{index + 1}"} for index, source in enumerate(deduplicate_sources(independent)[:MAX_SOURCES])]
    claims = []
    ranking_seconds, inference_seconds = 0.0, 0.0
    for index, claim in enumerate(article["claims"]):
        evidence = []
        hypothesis = resolve_claim(claim, article["claims"][:index])
        phase_started = time.perf_counter()
        ranked = rank_passages(hypothesis, sources)
        ranking_seconds += time.perf_counter() - phase_started
        for candidate in ranked:
            phase_started = time.perf_counter()
            scored = model.compare(candidate["passage"], hypothesis)
            inference_seconds += time.perf_counter() - phase_started
            evidence.append({**candidate, **scored})
        evidence = gate_contradictions(evidence, hypothesis)
        verdict, strength, reason = decide_claim(evidence, bool(sources))
        relaxed_used = None
        if verdict in {"insufficient_evidence", "conflicting"} and evidence:
            # Re-check only the core of the claim against the same passages.
            for variant in relaxed_claims(hypothesis):
                phase_started = time.perf_counter()
                core = gate_contradictions([{**row, **model.compare(row["passage"], variant)} for row in evidence], variant)
                inference_seconds += time.perf_counter() - phase_started
                core_verdict, _, _ = decide_claim(core, True)
                symmetric_contra = any(row["relation"] == "contradiction" for row in evidence)
                if core_verdict == "supported" and not symmetric_contra:
                    verdict, strength, relaxed_used = "supported", "partial", variant
                    reason = ("The core of this statement is supported by the sources, but some details "
                              "(exact wording, or who it was said to) differ from what the sources report.")
                    evidence = [{**row, "core_claim_check": {"claim": variant, "relation": c["relation"],
                                 "scores": c["scores"]}} for row, c in zip(evidence, core)]
                    break
        evidence.sort(key=lambda row: (row.get("core_claim_check", {}).get("relation") == "entailment", row["decision"] == "decisive", row["decision"] == "likely",
                                       row.get("raw_relation", row["relation"]) != "neutral", row["confidence"]), reverse=True)
        best = evidence[0] if evidence else None
        review_reason = ("no_sources" if not sources else "no_relevant_passages" if not evidence else "model_uncertain") if verdict == "insufficient_evidence" else None
        evidence_summary = {"sources_found": len(sources), "sources_compared": len({row["source_id"] for row in evidence}),
                            "passages_compared": len(evidence),
                            "decisive_support_sources": len({row["source_id"] for row in evidence if row["relation"] == "entailment"}),
                            "decisive_contradiction_sources": len({row["source_id"] for row in evidence if row["relation"] == "contradiction"})}
        if review_reason == "model_uncertain":
            label = "Evidence found; review required"
        elif strength == "partial":
            label = "Mostly true — wording/details differ"
        elif strength == "moderate" and verdict in {"supported", "contradicted"}:
            label = "Likely true (moderate confidence)" if verdict == "supported" else "Likely false (moderate confidence)"
        else:
            label = {"supported": "Supported — likely true", "contradicted": "Contradicted — likely false"}.get(verdict, LABELS[verdict])
        claims.append({"id": f"C{index + 1}", "text": claim, "resolved_text": hypothesis if hypothesis != claim else None, "core_claim_checked": relaxed_used,
                       "verdict": verdict, "strength": strength,
                       "label": label, "review_reason": review_reason, "evidence_summary": evidence_summary,
                       "scores": best["scores"] if best else None, "confidence": best["confidence"] if best else None,
                       "model_prediction": best.get("raw_relation", best["relation"]) if best else None,
                       "decision_reason": best.get("decision_reason", reason) if best else reason,
                       "reason": reason, "evidence": evidence})
    timings["ranking_ms"] = round(ranking_seconds * 1000)
    timings["inference_ms"] = round(inference_seconds * 1000)
    comparisons = sum(len(claim["evidence"]) for claim in claims)
    all_evidence = [row for claim in claims for row in claim["evidence"]]
    source_usage = {"sources_found": len(sources), "sources_compared": len({row["source_id"] for row in all_evidence}),
                    "passages_compared": comparisons,
                    "decisive_support_sources": len({row["source_id"] for row in all_evidence if row["relation"] == "entailment"}),
                    "decisive_contradiction_sources": len({row["source_id"] for row in all_evidence if row["relation"] == "contradiction"})}
    for source in sources:
        source["used_for_claims"] = [claim["id"] for claim in claims if any(row["source_id"] == source["id"] for row in claim["evidence"])]
    counts = {key: sum(row["verdict"] == key for row in claims) for key in LABELS}
    resolved = counts["supported"] + counts["contradicted"]
    if counts["supported"] and counts["contradicted"] or counts["conflicting"]:
        assessment = "mixed"
        label = ("Mixed — partly true, partly false" if counts["supported"] and counts["contradicted"] else "Mixed evidence")
    elif counts["contradicted"]:
        assessment, label = "contradicted", "Likely false — contradicted by sources"
    elif counts["supported"] and counts["supported"] >= max(1, (len(claims) + 1) // 2):
        assessment = "supported"
        partial = any(row["strength"] == "partial" for row in claims)
        label = ("Likely true — supported by sources" if counts["supported"] == len(claims) and not partial
                 else "Mostly true — key claims supported")
    elif counts["supported"]:
        assessment, label = "insufficient_evidence", "Partly verified — most claims unconfirmed"
    else:
        assessment, label = "insufficient_evidence", "Unverified — model needs review" if any(row["evidence"] for row in claims) else "Unverified — more evidence needed"
        if not sources and search["status"] == "unavailable":
            label = "Search unavailable"
    summary = f"{counts['supported']} of {len(claims)} checked statements are supported; {counts['contradicted']} contradicted; {counts['insufficient_evidence']} unresolved."
    if counts["conflicting"]:
        summary += f" {counts['conflicting']} have conflicting evidence."
    moderate = sum(row["strength"] in {"moderate", "partial"} for row in claims if row["verdict"] in {"supported", "contradicted"})
    if moderate:
        summary += f" {moderate} of the {resolved} resolved verdicts are moderate-confidence."
    limitations = ["Support means agreement with the cited passage, not proof that the source or whole article is true.",
                   "Search covers a selection of sources. Copied reporting is not independent corroboration.",
                   "The model can miss numerical changes, negation, satire and context. Open the citations before relying on a result."]
    if article["omitted"]:
        limitations.append(f"Only the first {len(claims)} eligible statements were checked; {article['omitted']} other text segments were not checked.")
    if any(item["truncated"] for row in claims for item in row["evidence"]):
        limitations.append("Some evidence/claim pairs exceeded the model's token window and were truncated.")
    return {"assessment": assessment, "label": label, "summary": summary, "claims": claims,
            "sources": [{key: value for key, value in source.items() if key != "text"} for source in sources],
            "coverage": {"total_sentences": article["total_sentences"], "checked_claims": len(claims), **counts, "omitted": article["omitted"]},
            "search": search, "warnings": list(dict.fromkeys(warnings)), "word_count": len(re.findall(r"\b[\w'-]+\b", text)),
            "source_usage": source_usage, "timings": timings, "comparisons": comparisons,
            "model": {"name": model.config["model_name"], "sha256": model.config["sha256"], "max_tokens": model.max_tokens},
            "latency_ms": round((time.perf_counter() - started) * 1000), "limitations": limitations}
