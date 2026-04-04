from pydantic import BaseModel
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.3

class GenerateResponse(BaseModel):
    response: str
    ttt: bool

class CompareResponse(BaseModel):
    prompt: str
    ttt_on: str
    ttt_off: str

class ResetResponse(BaseModel):
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str