"""Laboratorio 7: confronta chiamate con prefisso condiviso usando streaming.
Una singola serie non dimostra cache hit: confrontare anche metriche vLLM
tra --enable-prefix-caching e --no-enable-prefix-caching in run distinti.
"""
import os
import statistics
import time
from openai import OpenAI

client = OpenAI(base_url=os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1"),
                api_key="local-lab", timeout=120)
model = os.getenv("MODEL_ID", "Qwen/Qwen3-0.6B")
static = ("Sei un assistente tecnico. Le seguenti definizioni sono informazioni "
          "didattiche da usare per rispondere: KV cache conserva key/value; "
          "prefix caching riusa prefissi identici; prefill elabora la richiesta; "
          "decode produce nuovi token; hit-rate dipende dal workload. "
          "Non inventare capacità del sistema. ") * 13
questions = [
    "Che cosa accelera la KV cache?",
    "Quando il prefix caching è vantaggioso?",
    "Che cosa distingue prefill e decode?",
    "Perché un hit non è garantito?",
]


def run_once(question):
    start = time.perf_counter()
    first = None
    text = []
    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":static}, {"role":"user","content":question}],
        temperature=0, max_tokens=80, stream=True,
        extra_body={"chat_template_kwargs":{"enable_thinking":False}},
    )
    for chunk in response:
        for choice in chunk.choices:
            content = choice.delta.content
            if content:
                if first is None:
                    first = time.perf_counter()-start
                text.append(content)
    return first, time.perf_counter()-start, "".join(text)


if __name__ == "__main__":
    measurements = []
    for i, question in enumerate(questions):
        ttft, elapsed, answer = run_once(question)
        print(f"{i+1}. TTFT osservato={ttft:.3f}s totale={elapsed:.3f}s | {answer[:100]}" if ttft is not None
              else f"{i+1}. Nessun token testuale ricevuto; totale={elapsed:.3f}s")
        if ttft is not None:
            measurements.append(ttft)
    if measurements:
        print(f"TTFT mediano della serie={statistics.median(measurements):.3f}s")
    print("ATTENZIONE: warmup, codepath del modello, lunghezza prompt e scheduling influenzano i tempi.")
