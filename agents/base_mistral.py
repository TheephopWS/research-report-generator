import os
from abc import ABC, abstractmethod
from typing import Dict, Any

from mistralai import Mistral

MODEL = "mistral-small-latest"

class BaseAgent(ABC):
    id: str
    start_message: str

    def __init__(self):
        self.client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

    async def call_mistral(
        self,
        system: str,
        user: str,
        max_tokens: int = 1500,
    ) -> str:
        response = await self.client.chat.complete_async(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    @abstractmethod
    async def run(self, context: Dict[str, Any]) -> str:
        pass
