"""DeepEval's built-in metrics default to a paid OpenAI judge. This wraps
Groq's gpt-oss-20b instead (plan.md Decision #5: free, and a different model
family from both assistants, which avoids self-preference bias).
"""
from __future__ import annotations

from deepeval.models.base_model import DeepEvalBaseLLM


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = "groq/gpt-oss-20b", api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        import litellm

        r = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "groq-gpt-oss-20b"
