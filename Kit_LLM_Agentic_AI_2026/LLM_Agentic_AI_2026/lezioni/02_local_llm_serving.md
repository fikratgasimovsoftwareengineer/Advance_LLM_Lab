# Modulo 2 — Local LLM & Model Serving
**Durata:** 4 ore.

## 2.1 Esecuzione locale: componenti
Un'installazione locale separa quattro livelli: (1) i pesi/tokenizer, (2) il runtime di inferenza, (3) il server HTTP, (4) il client. Nel tuo caso: `Qwen/Qwen3-0.6B` → vLLM → `/v1/chat/completions` → `curl` o client OpenAI. L'API compatibile consente di cambiare spesso il backend senza riscrivere tutta l'applicazione, ma non garantisce supporto identico a ogni parametro proprietario.

**Transformers** è una libreria generale per caricare e addestrare/eseguire modelli; **Ollama** semplifica packaging e uso locale; **vLLM** ottimizza il serving concorrente sfruttando gestione della KV cache, batching continuo e pianificazione delle richieste. **TGI** è un'altra opzione per il serving; la scelta dipende da hardware, modello, opzioni supportate, traffico, monitoring e deployment.

## 2.2 VRAM: non bastano i parametri
I pesi in BF16/FP16 richiedono in prima approssimazione `2 × N` byte (con N parametri). Per 8 miliardi di parametri, circa 16 GB **solo di pesi**. A questi si aggiungono KV cache, buffer, activations, allocazioni del framework, prompt lunghi e richieste concorrenti. Quantizzare i pesi a 8 o 4 bit riduce tale componente, con overhead e possibili compromessi in qualità; non elimina automaticamente la KV cache.

Misure richieste: `nvidia-smi`, VRAM libera/occupata, throughput token/s, TTFT (time-to-first-token), latenza end-to-end, TPOT (time per output token), P50/P95/P99, error rate e concurrency. Distinguere throughput aggregato da latenza individuale: l'aumento di throughput può peggiorare P95.

## 2.3 Prefill, decode, KV cache, batching
**Prefill** elabora il prompt e popola Key/Value per i token del contesto. **Decode** genera token successivi riutilizzando le rappresentazioni memorizzate. La **KV cache** evita di ricalcolare tutta la storia a ogni token, ma consuma memoria GPU. Il **continuous batching** permette di aggiungere/togliere richieste senza aspettare che l'intero gruppo termini; ciò migliora l'uso della GPU. **PagedAttention** gestisce blocchi di memoria per le rappresentazioni KV in modo più efficiente.

Le impostazioni di scheduling, parallelismo e caching variano con la versione di vLLM. Usare sempre `vllm serve --help` e `vllm --version` sul server reale prima di copiare opzioni da una guida esterna.

## 2.4 Laboratorio consigliato
```bash
# Nel terminale server (ambiente vLLM preesistente)
vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000 \
  --dtype auto --max-model-len 2048 --gpu-memory-utilization 0.75

# Nel terminale client
curl -s http://127.0.0.1:8000/v1/models
python laboratori/02_benchmark_concorrenza.py --concurrency 1 --requests 12
python laboratori/02_benchmark_concorrenza.py --concurrency 5 --requests 20
```
Confronta la latenza mediana, la P95, il numero di completamenti riusciti e il throughput approssimato. Per un benchmark più rigoroso verifica `vllm bench serve --help` e fissa la lunghezza dell'input e output.

## 2.5 Produzione
Esporre `127.0.0.1` consente accesso locale; usare `0.0.0.0` rende la porta potenzialmente raggiungibile dalla rete. Prima di esporre un server LLM: autenticazione, TLS in reverse proxy, rate limiting, limiti per utente, log privi di contenuti sensibili, timeouts, quote e monitoraggio della GPU. Docker non equivale a isolamento di sicurezza completo. In Kubernetes: Deployment, Service, resource limits GPU, readiness probe, autoscaling realistico e ingress autenticato; non pubblicare direttamente un server senza protezioni.

## Autoverifica
1. Come distingui KV cache e pesi?
2. Perché 50 richieste concorrenti non implicano 50 modelli in VRAM?
3. Cosa misureresti per scegliere tra Ollama e vLLM?
4. Perché una GPU con 16 GB non garantisce che un 8B FP16 funzioni con prompt lunghi?
5. Che cosa cambia tra `host 127.0.0.1` e `host 0.0.0.0`?

## Video suggeriti
https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm
https://www.deeplearning.ai/courses/efficiently-serving-llms
