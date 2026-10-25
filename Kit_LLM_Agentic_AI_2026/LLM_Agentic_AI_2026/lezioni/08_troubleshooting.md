# Modulo 8 — Troubleshooting, osservabilità e quality assurance
**Durata:** 3 ore.

## 8.1 Diagnostica per livelli
Un errore end-to-end va localizzato per componente, senza modificare tutto insieme:
1. **Host/GPU**: `uname -m`, `nvidia-smi`, driver, VRAM disponibile, processi residenti.
2. **Runtime**: ambiente Python attivo, `which vllm`, `vllm --version`, PyTorch/CUDA compatibili.
3. **Model loading**: identificativo e accessibilità del modello, tokenizer/chat template, dimensione della context window.
4. **Serving HTTP**: bind host, porta occupata, readiness endpoint, request schema, errori 400/401/404/429/500.
5. **Applicazione**: Pydantic/JSON parsing, prompt injection, fallimenti di tool, timeout/retry.
6. **Qualità**: retrieval errato, chunk non pertinenti, allucinazioni, citazioni difettose, regressioni del modello.

## 8.2 Problemi incontrati nell'esercitazione
`vllm download-kernels` non riconosciuto: **subcommand assente** in quella CLI. Prima di usare esempi da Internet, verificare `vllm --help`, non reinstallare alla cieca. `finish_reason=length`: risposta interrotta al limite `max_tokens`; aumentare solo dopo aver controllato VRAM, prompt length e comportamento del modello. Marcatori `<think>` in `message.content`: comportamento legato a Qwen3/template/parametri, non necessariamente un crash; usare il parametro per disabilitare thinking quando supportato.

`CUDA out of memory`: verificare disponibilità effettiva, `gpu_memory_utilization`, `max_model_len`, quantizzazione, batch/concurrency e processi concorrenti. Cambiare i parametri uno per volta e salvare benchmark. `Unable to locate credentials` con boto3: verificare identity/credential provider chain **nell'ambiente in cui gira il processo**, non solo sull'host. `Access to model ... not available`: può essere questione di permessi, region, entitlement o tipo di account; verificare con i servizi AWS e gli identificativi corretti del modello.

## 8.3 Metriche e tracing
Registrare request ID e correlation ID tra frontend, backend, agente, vector DB e model server. Per ogni chiamata: modello, versione, token input/output, dimensione contesto, numero di retrieval, query rewriting, chiamate tool, errori, latenza, TTFT, cache hit (se disponibile). Evitare di salvare prompt privati nei log senza basi legali e protezioni. Le metriche aggregate non sostituiscono trace diagnostiche con cause/contesto.

## 8.4 Strategia di test
**Unit test** di parser, routing e tool; **integration test** di RAG/model server; **eval offline** con dataset versionato; **load test** con concorrenza e rate limit; **security test** con prompt injection; **canary** e rollback in produzione. Non trattare un singolo output convincente come misura statistica.

## Esercizio
Lancia `laboratori/08_diagnostica.py`; simula una porta errata per verificare la gestione degli errori; ripeti con il server attivo. Crea una scheda incidente con "sintomo, ipotesi, prova, esito, correzione, regressione".

## Autoverifica
1. Come distingui un errore CUDA da un errore applicativo HTTP?
2. Quali evidenze raccogli per un errore 500 intermittente?
3. Che differenza c'è tra log, metriche e trace?
4. Come dimostri che una modifica al retrieval ha migliorato la qualità?
5. Perché la procedura di rollback fa parte del troubleshooting?
