# Videolezione 07 — Prompt caching e prestazioni
**Durata suggerita:** 9-12 minuti. **Riprese:** timeline prefill/decode, demo stream e due esecuzioni server.

**00:00-03:00 — Il vocabolario delle cache.**
«KV cache, prefix caching e semantic response cache sono tre strumenti diversi. La KV cache mantiene rappresentazioni nel corso dell'inferenza. Prefix caching può riusare il prefill tra richieste con un prefisso identico. Semantic caching restituisce una risposta già pronta, con rischi di obsolescenza e isolamento dei dati.»

**03:00-06:00 — Prefissi identici.**
Mostra due prompt con system prompt uguale e due domande diverse. «Se cambia la prima parte, il riuso diminuisce. Un prefix cache hit accelera soprattutto l'elaborazione del prompt, non automaticamente la generazione dei nuovi token.»

**06:00-09:00 — Demo.**
Esegui `07_prefix_cache_lab.py`. Mostra i TTFT, senza promettere speedup fissi. Riavvia il server, se possibile, con il parametro di caching disattivato/attivato e ripeti prove. «Non basta osservare un tempo inferiore: bisogna controllare misure ripetute e metriche di cache hit.»

**09:00-12:00 — Bedrock.**
«Il caching dei provider gestiti usa condizioni e prezzi specifici del modello. Un prompt statico lungo può essere memorizzato, ma TTL, token minimi e costo cache-write vanno verificati. Il criterio non è avere cache, è ridurre davvero costi e latenza.»

**Risorsa:** https://www.deeplearning.ai/courses/building-toward-computer-use-with-anthropic
