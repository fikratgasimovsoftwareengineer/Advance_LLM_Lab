"""Laboratorio 1: chiamare Qwen via server vLLM e leggere l'usage.
Avvio: python 01_inferenza_client.py (server già attivo su localhost:8000).
"""
import os
import time
from openai import OpenAI

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
client = OpenAI(base_url=os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1"), api_key="local-lab", timeout=90)

for max_tokens in (64, 400):
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Rispondi in italiano con frasi semplici."},
            {"role": "user", "content": "Spiega la differenza tra un token e una parola in 3 frasi."},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    seconds = time.perf_counter() - start
    choice = response.choices[0]
    print(f"\nmax_tokens={max_tokens} | {seconds:.2f}s | finish_reason={choice.finish_reason}")
    print(f"usage={response.usage}")
    print(choice.message.content)
