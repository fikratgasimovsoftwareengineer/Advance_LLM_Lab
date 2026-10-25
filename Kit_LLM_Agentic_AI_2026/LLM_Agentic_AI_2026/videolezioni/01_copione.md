# Videolezione 01 — Dentro un LLM: dal prompt al token
**Durata suggerita:** 8-10 minuti. **Formato:** screencast 16:9, terminale e slide. **Obiettivo:** distinguere token, Transformer, sampling e fine del completamento.

**00:00-01:00 — Apertura, inquadratura del terminale.**
Voce: «Un LLM non cerca una frase pronta in un archivio. Data una sequenza di token, stima quali token possano seguire. Per questo è in grado di generare testo, ma può anche costruire frasi fluenti non corrette. Oggi useremo un modello che abbiamo già eseguito in locale: Qwen3-0.6B, accessibile tramite vLLM.»

**01:00-03:00 — Slide “tokenizer → embedding → Transformer → logits → token”.**
Voce: «Il testo passa prima dal tokenizer, che lo spezza in unità numeriche. Le rappresentazioni vengono trasformate dai blocchi Transformer. L'attenzione permette a ogni posizione di combinare informazioni dal contesto disponibile, nel rispetto della causal mask. Infine il modello produce punteggi per i possibili token successivi: i logits.»

**03:00-05:00 — Schermo diviso: temperatura e `max_tokens`.**
Voce: «I parametri di campionamento non cambiano ciò che è stato appreso. Cambiano il modo di scegliere il prossimo token. Con `max_tokens=64` possiamo troncare una spiegazione corretta a metà frase. Il nostro precedente output mostrava proprio `finish_reason=length`: il server funzionava, era soltanto finito il budget di generazione.»

**05:00-08:00 — Demo.**
Mostra `curl localhost:8000/v1/models`; apri `laboratori/01_inferenza_client.py` ed eseguilo. Evidenzia `usage.prompt_tokens`, `usage.completion_tokens` e `finish_reason`. Spiega che il prefisso `<think>` apparso in un esperimento dipende dalla modalità del modello e dal template.

**08:00-10:00 — Chiusura e verifica.**
Domanda orale: «Perché 256 token in output non significano 256 parole? Che cosa posso inferire da una risposta troncata?». Assegna esercizio: cambiare un solo parametro e annotare la differenza, senza attribuire variazioni al caso sbagliato.

**Risorsa esterna:** https://www.deeplearning.ai/courses/how-transformer-llms-work
