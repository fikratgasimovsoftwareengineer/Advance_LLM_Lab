"""Laboratorio 2: semplice prova di carico locale, NON benchmark scientifico.
Esempio: python 02_benchmark_concorrenza.py --concurrency 5 --requests 20
"""
import argparse
import concurrent.futures
import os
import statistics
import time
from openai import OpenAI

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--concurrency", type=int, default=3)
PARSER.add_argument("--requests", type=int, default=12)
ARGS = PARSER.parse_args()
if not 1 <= ARGS.concurrency <= 20 or not 1 <= ARGS.requests <= 200:
    PARSER.error("Questo laboratorio permette 1..20 thread e 1..200 richieste")
MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")


def one_request(i):
    client = OpenAI(base_url=URL, api_key="local-lab", timeout=120)
    start = time.perf_counter()
    try:
        answer = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user", "content":f"In una frase spiega il concetto AI numero {i % 4 + 1}."}],
            max_tokens=70, temperature=0.0,
            extra_body={"chat_template_kwargs":{"enable_thinking":False}},
        )
        return {"ok": True, "seconds": time.perf_counter()-start,
                "tokens": answer.usage.completion_tokens if answer.usage else 0}
    except Exception as exc:
        return {"ok": False, "seconds": time.perf_counter()-start, "error": str(exc)[:180], "tokens":0}


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    index = (len(values)-1) * pct/100
    low = int(index)
    high = min(low+1, len(values)-1)
    return values[low] + (values[high] - values[low]) * (index-low)


if __name__ == "__main__":
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=ARGS.concurrency) as pool:
        results = list(pool.map(one_request, range(ARGS.requests)))
    elapsed = time.perf_counter()-started
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    latencies = [r["seconds"] for r in ok]
    print(f"concorrenza={ARGS.concurrency} richieste={ARGS.requests} riuscite={len(ok)} errori={len(bad)}")
    print(f"durata globale={elapsed:.2f}s throughput richieste/s={len(ok)/elapsed:.2f}")
    if ok:
        print(f"latenza mediana={statistics.median(latencies):.2f}s P95={percentile(latencies,95):.2f}s")
        print(f"token output/s (complessivi, approssimati)={sum(r['tokens'] for r in ok)/elapsed:.2f}")
    for err in bad[:3]:
        print("errore:", err["error"])
