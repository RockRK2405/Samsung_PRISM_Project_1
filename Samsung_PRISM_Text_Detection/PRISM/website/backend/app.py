"""FastAPI application for the PRISM AI Text Detection Platform.

Serves the frontend static files and provides API endpoints for
text analysis, batch processing, and model metrics.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import io
import docx
import PyPDF2

# Ensure the project src is on the path
_project_root = Path(__file__).resolve().parent.parent.parent
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from website.backend.model_loader import get_ensemble, get_metrics

# ======================================================================
# App setup
# ======================================================================
app = FastAPI(
    title="PRISM AI Text Detector",
    description="Multi-signal AI-generated text detection with explainability",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/css", StaticFiles(directory=str(_frontend_dir / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(_frontend_dir / "js")), name="js")
    if (_frontend_dir / "assets").exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_frontend_dir / "assets")),
            name="assets",
        )


# ======================================================================
# Pydantic models
# ======================================================================
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyse")


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of texts to analyse")


# ======================================================================
# Frontend routes
# ======================================================================
@app.get("/")
async def serve_index():
    """Serve the landing page."""
    index = _frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "Frontend not built. Place files in website/frontend/"})


@app.get("/analyzer")
@app.get("/analyzer.html")
async def serve_analyzer():
    f = _frontend_dir / "analyzer.html"
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse({"error": "analyzer.html not found"}, status_code=404)


@app.get("/batch")
@app.get("/batch.html")
async def serve_batch():
    f = _frontend_dir / "batch.html"
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse({"error": "batch.html not found"}, status_code=404)


@app.get("/dashboard")
@app.get("/dashboard.html")
async def serve_dashboard():
    f = _frontend_dir / "dashboard.html"
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse({"error": "dashboard.html not found"}, status_code=404)


@app.get("/about")
@app.get("/about.html")
async def serve_about():
    f = _frontend_dir / "about.html"
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse({"error": "about.html not found"}, status_code=404)


# ======================================================================
# API endpoints
# ======================================================================
@app.get("/api/health")
async def health_check():
    """Health check and model status."""
    ensemble = get_ensemble()
    return {
        "status": "ok",
        "model_loaded": ensemble.stylometric_model is not None,
        "perplexity_available": ensemble.weights.get("perplexity", 0) > 0,
    }


@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest):
    """Analyse a single block of text."""
    ensemble = get_ensemble()
    if not ensemble:
        raise HTTPException(status_code=500, detail="Ensemble model failed to load")

    try:
        result = ensemble.analyze(request.text)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def analyze_file(file: UploadFile = File(...)):
    """Extract text from an uploaded file and analyze it."""
    ensemble = get_ensemble()
    if not ensemble:
        raise HTTPException(status_code=500, detail="Ensemble model failed to load")

    try:
        content = await file.read()
        filename = file.filename.lower()
        extracted_text = ""

        if filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            extracted_text = "\n".join([p.text for p in doc.paragraphs])
        elif filename.endswith(".txt"):
            extracted_text = content.decode("utf-8", errors="replace")
        elif filename.endswith((".jpg", ".jpeg", ".png")):
            # Placeholder for Image Analysis
            return {
                "verdict": "Model Pending Integration",
                "confidence": "N/A",
                "authenticity_score": 50.0,
                "explanation": "Image uploaded successfully. The computer vision AI module is pending integration by the team.",
                "signals": {},
                "linguistic_flags": [],
                "sentences": [],
                "features": {},
                "media_type": "image",
                "filename": filename
            }
        elif filename.endswith((".mp3", ".wav")):
            # Placeholder for Audio Analysis
            return {
                "verdict": "Model Pending Integration",
                "confidence": "N/A",
                "authenticity_score": 50.0,
                "explanation": "Audio uploaded successfully. The audio/speech AI module is pending integration by the team.",
                "signals": {},
                "linguistic_flags": [],
                "sentences": [],
                "features": {},
                "media_type": "audio",
                "filename": filename
            }
        elif filename.endswith(".mp4"):
            # Placeholder for Video Analysis
            return {
                "verdict": "Model Pending Integration",
                "confidence": "N/A",
                "authenticity_score": 50.0,
                "explanation": "Video uploaded successfully. The video analysis AI module is pending integration by the team.",
                "signals": {},
                "linguistic_flags": [],
                "sentences": [],
                "features": {},
                "media_type": "video",
                "filename": filename
            }
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF, DOCX, TXT, JPG, PNG, MP3, WAV, or MP4.")

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the file.")

        # Cap text length to prevent memory overload (e.g., 200,000 characters)
        if len(extracted_text) > 200000:
            extracted_text = extracted_text[:200000]

        result = ensemble.analyze(extracted_text)
        # Include extracted text snippet so frontend can display it
        result["extracted_text"] = extracted_text
        result["media_type"] = "text"
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/api/analyze/batch")
async def analyze_batch(request: BatchAnalyzeRequest):
    """Analyse multiple texts. Returns results list and summary stats."""
    try:
        ensemble = get_ensemble()
        result = ensemble.analyze_batch(request.texts)
        return result
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/metrics")
async def model_metrics():
    """Return model performance metrics and info."""
    raw_metrics = get_metrics()
    ensemble = get_ensemble()

    # Build a structured response
    tp = raw_metrics.get("true_positive", 0)
    fp = raw_metrics.get("false_positive", 0)
    fn = raw_metrics.get("false_negative", 0)
    tn = raw_metrics.get("true_negative", 0)

    return {
        "accuracy": raw_metrics.get("accuracy", 0),
        "f1": raw_metrics.get("f1_synthetic", 0),
        "roc_auc": raw_metrics.get("roc_auc", 0),
        "fpr": raw_metrics.get("false_positive_rate_real_flagged_synthetic", 0),
        "precision": raw_metrics.get("precision_synthetic", 0),
        "recall": raw_metrics.get("recall_synthetic", 0),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "model_info": {
            "version": "2.0.0",
            "mode": raw_metrics.get("mode", "unknown"),
            "features_count": 286,
            "classifier": "Multi-Signal Ensemble",
            "total_rows": raw_metrics.get("total_rows", 0),
            "train_rows": raw_metrics.get("train_rows", 0),
            "threshold": raw_metrics.get("threshold", 0.5),
        },
        "recent_predictions": [],  # Populated at runtime if needed
    }


@app.get("/api/model-info")
async def model_info():
    """Return detailed model information."""
    ensemble = get_ensemble()
    return {
        "version": "2.0.0",
        "name": "PRISM Multi-Signal Ensemble",
        "signals": [
            {
                "name": "Stylometric Analysis",
                "type": "TF-IDF + Advanced Features + Classifier",
                "weight": ensemble.weights.get("stylometric", 0),
            },
            {
                "name": "Perplexity Analysis",
                "type": "DistilGPT-2 Surprisal Diversity",
                "weight": ensemble.weights.get("perplexity", 0),
            },
            {
                "name": "Linguistic Pattern Detection",
                "type": "Rule-based Pattern Analysis",
                "weight": ensemble.weights.get("linguistic", 0),
            },
        ],
        "threshold": ensemble.threshold,
        "model_loaded": ensemble.stylometric_model is not None,
    }
