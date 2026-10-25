# Videolezione 08 — Troubleshooting di un sistema LLM
**Durata suggerita:** 9-12 minuti. **Riprese:** shell e log in zoom, scheda incident.

**00:00-03:00 — Diagnosi per livelli.**
«Quando una risposta fallisce non bisogna reinstallare tutto. Separiamo GPU, runtime, model loading, endpoint REST, orchestrazione, retrieval e qualità della generazione. Per ciascun livello cerchiamo un'osservazione falsificabile.»

**03:00-05:00 — Un errore realmente visto.**
«Il comando `vllm download-kernels` è stato respinto dalla CLI: ciò non implica server rotto. `vllm --version` e `vllm --help` sono strumenti per riconciliare una guida con la versione installata.»

**05:00-08:00 — Errore vs risposta incompleta.**
«Un `finish_reason=length` indica il limite token, non un errore HTTP. Un `CUDA OOM` richiede analizzare VRAM, contesto e concorrenza. Un errore di credenziali Bedrock deve essere verificato dentro il container o processo dove gira il codice, non soltanto nel terminale dell'host.»

**08:00-12:00 — Esercizio.**
Esegui `08_diagnostica.py` prima a server attivo, poi con porta volutamente errata tramite `VLLM_URL=http://127.0.0.1:9999/v1`. Crea un incident report: sintomo, ipotesi, esperimento, correzione, test regressione.
