import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    GenerateRequest,
    GenerateResponse,
    CompareResponse,
    ResetResponse,
    HealthResponse
)
from api.model_loader import mac_model

app = FastAPI(
    title="MAC Language Model API",
    description="Memory-Augmented Context Language Model with Test-Time Training",
    version="1.0.0"
)

# CORS — allows UI to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    # load model when API starts
    checkpoint = os.environ.get("CHECKPOINT_PATH", "checkpoints/mac_best.pt")
    mac_model.load(checkpoint_path=checkpoint)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=mac_model.loaded,
        device=mac_model.device or "not loaded"
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if not mac_model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    response = mac_model.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        do_ttt=True
    )

    return GenerateResponse(response=response, ttt=True)


@app.post("/compare", response_model=CompareResponse)
async def compare(request: GenerateRequest):
    if not mac_model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ttt_on, ttt_off = mac_model.compare(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty
    )

    return CompareResponse(
        prompt=request.prompt,
        ttt_on=ttt_on,
        ttt_off=ttt_off
    )


@app.post("/reset", response_model=ResetResponse)
async def reset():
    if not mac_model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    mac_model.reset()
    return ResetResponse(message="Memory reset. Fresh conversation started.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)