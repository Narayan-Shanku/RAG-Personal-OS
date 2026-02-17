from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import httpx


@dataclass
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.2


class OllamaClient:
    def __init__(self, cfg: Optional[OllamaConfig] = None):
        self.cfg = cfg or OllamaConfig()

    async def generate(self, prompt: str) -> str:
        url = f"{self.cfg.base_url}/api/generate"
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()