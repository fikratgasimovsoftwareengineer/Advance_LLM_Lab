# Videolezione 02 — Servire un modello locale con vLLM
**Durata suggerita:** 12-15 minuti. **Riprese:** desktop Linux, `nvidia-smi`, terminali affiancati e un semplice grafico concorrenza/latency.

**00:00-02:00 — Il sistema reale.**
Voce: «Abbiamo installato vLLM in un ambiente virtuale Linux e verificato una risposta di Qwen3-0.6B. Mostro dove si trovano il modello, il server e il client. Il nome della GPU non va indovinato: lo ricaviamo dai dati di `nvidia-smi`. Il confronto con 7B, 8B e 32B dipende da VRAM e quantizzazione.»

**02:00-04:00 — Memoria e throughput.**
Mostra due rettangoli “pesi” e “KV cache”. Voce: «I pesi restano caricati, mentre la KV cache cresce con prompt, contesto e richieste. La memoria richiesta non è il semplice numero di parametri. Aumentare la lunghezza massima del contesto consuma capacità che potrebbe servire alla concorrenza.»

**04:00-06:00 — Prefill e decode.**
Voce: «Il prefill processa il contesto; il decode genera un token alla volta. Continuous batching permette di far entrare nuove richieste nel motore mentre altre sono già in fase di generazione. PagedAttention riduce sprechi nella gestione della cache.»

**06:00-11:00 — Laboratorio reale.**
Esegui `vllm --version`, `curl /v1/models`, quindi `python laboratori/02_benchmark_concorrenza.py --concurrency 1 --requests 12` e `--concurrency 5 --requests 20`. Annota su schermo mediane e P95 senza inventare valori. Spiega perché l'aumento di richieste può migliorare throughput ma peggiorare latenza.

**11:00-15:00 — Mini colloquio.**
«Quale server hai usato? Quale modello? Su quale GPU? Quali metriche hai misurato?». Distinguere un test sul proprio PC dal serving in produzione.

**Risorsa:** https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm
