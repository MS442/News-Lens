"""News Lens v2: Python source retrieval and trained evidence comparison."""
from __future__ import annotations
import json
import re
import threading
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from evidence_sources import SourceError, extract_article, public_url

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="News Lens", version="2.1.0", docs_url="/api/docs", redoc_url=None)
_classifier = None
_load_lock = threading.Lock()
_checks = threading.BoundedSemaphore(2)

class Article(BaseModel):
    text: str = Field(min_length=20, max_length=30000)
    source_url: str | None = Field(default=None, max_length=2048)
    evidence_text: str | None = Field(default=None, max_length=30000)
    evidence_url: str | None = Field(default=None, max_length=2048)
    search_web: bool = True

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value):
        value = value.strip()
        if re.fullmatch(r"https?://\S+", value):
            raise ValueError("Use Article link to load the page before checking it.")
        words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", value)
        letters = [char for char in value if char.isalpha()]
        if len(words) < 5 or len(set(word.lower() for word in words)) < 4:
            raise ValueError("Paste a complete factual sentence, not an empty or repeated fragment.")
        if sum(char.isascii() for char in letters) / max(1, len(letters)) < .85:
            raise ValueError("This model supports English text. Please use an English article.")
        return value

    @field_validator("source_url", "evidence_url")
    @classmethod
    def valid_link(cls, value):
        return public_url(value.strip(), resolve=False)[0] if value and value.strip() else None

    @field_validator("evidence_text")
    @classmethod
    def strip_evidence(cls, value):
        return (value.strip() or None) if value else None

class ArticleLink(BaseModel):
    url: str = Field(min_length=8, max_length=2048)

def classifier():
    global _classifier
    if _classifier is None:
        with _load_lock:
            if _classifier is None:
                if not (ROOT / "models/evidence/runtime_config.json").exists():
                    raise HTTPException(503, "The evidence model is still being trained. Predictions will appear after evaluation.")
                from evidence_model import EvidenceClassifier
                try:
                    _classifier = EvidenceClassifier()
                except (RuntimeError, OSError, ValueError, KeyError) as error:
                    raise HTTPException(503, str(error) if isinstance(error, RuntimeError) else "The evidence model is temporarily unavailable.") from error
    return _classifier

@app.middleware("http")
async def response_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response

@app.get("/api/health")
def health():
    try:
        model = classifier()
    except HTTPException as error:
        return JSONResponse({"status": "unavailable", "detail": error.detail}, status_code=503)
    return {"status": "ready", "model_loaded": True, "runtime_self_test": "passed",
            "inference": "Python ONNX Runtime", "external_classification_api": False, "external_search": "Exa",
            "model_sha256": model.config["sha256"], "version": "2.1.0"}

@app.post("/api/extract")
def extract(link: ArticleLink):
    try:
        return extract_article(link.url.strip())
    except SourceError as error:
        raise HTTPException(422, str(error)) from error

@app.post("/api/predict")
def predict(article: Article):
    request_started = time.perf_counter()
    model = classifier()
    model_loading_ms = round((time.perf_counter() - request_started) * 1000)
    if not _checks.acquire(blocking=False):
        raise HTTPException(429, "Two checks are already running. Please try again shortly.")
    try:
        from evidence_pipeline import analyse_article
        result = analyse_article(model, **article.model_dump())
        result.setdefault("timings", {}).update({
            "model_loading_ms": model_loading_ms,
            # Measures the handler; platform startup and network transit are external.
            "handler_total_ms": round((time.perf_counter() - request_started) * 1000),
        })
        return result
    except (SourceError, ValueError) as error:
        raise HTTPException(422, str(error)) from error
    finally:
        _checks.release()

@app.get("/api/metrics")
def metrics(download: bool = False):
    path = ROOT / "artifacts/evidence_metrics.json"
    config_path = ROOT / "models/evidence/runtime_config.json"
    if not path.exists() or not config_path.exists():
        raise HTTPException(503, "Evidence model evaluation is still running. Previous article-classification scores are not substituted.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("evaluation_pending"):
        raise HTTPException(503, "Evaluation is still running; incomplete metrics are withheld.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["model"]["sha256"] != config["sha256"]:
        raise HTTPException(503, "The evaluation does not match the deployed model.")
    if download:
        return FileResponse(path, media_type="application/json", filename="news-lens-evidence-evaluation.json")
    return JSONResponse(report)

@app.get("/api/metrics/legacy")
def legacy_metrics():
    return {"scope": "Historical article-pattern classifier; not used for current evidence verdicts.",
            "report": json.loads((ROOT / "artifacts/metrics.json").read_text(encoding="utf-8"))}

@app.get("/api/examples")
def examples():
    path = ROOT / "artifacts/evidence_examples.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

@app.get("/")
def index():
    return FileResponse(ROOT / "public/index.html")

# Vercel serves public/ on its CDN and removes it from the function bundle.
if (ROOT / "public").is_dir():
    app.mount("/static", StaticFiles(directory=ROOT / "public"), name="static")

