# Modulo 7 — Prompt Caching, Prefix Caching, KV Cache e costi
**Durata:** 3 ore.

## 7.1 Tre cache diverse
1. **KV cache (inference-time state)**: Key/Value per i token già processati durante la generazione corrente. Evita di ricalcolare la storia a ogni token, ma occupa VRAM.
2. **Prefix/prompt caching**: riusa parte della computazione di prefill tra richieste diverse che condividono lo stesso **prefisso di token**. Alcuni fornitori gestiti fatturano in modo specifico cache write/cache read; le regole dipendono dal modello.
3. **Semantic response caching**: può restituire una risposta memorizzata per domande simili, evitando del tutto un'inferenza. È rischiosa se i dati sono personalizzati, cambiano rapidamente o richiedono citazioni contestuali.

Il *caching di risultati dei tool* è ancora un quarto problema: bisogna conservare validità temporale, permessi e versione dei dati. Non usare un risultato cached tra tenant diversi senza una politica di isolamento ben progettata.

## 7.2 Automatic Prefix Caching in vLLM
vLLM può riutilizzare blocchi di KV cache già calcolati quando due richieste hanno un prefisso comune; su vLLM 0.31.0 l'opzione `enable_prefix_caching` esiste. Non dedurre che un hit si verifichi ogni volta: dimensione del prefisso, disponibilità nella cache, politiche di eviction e tokenizzazione influiscono. Il vantaggio riguarda **prefill**, non accelera direttamente la generazione dei token nuovi.

Per il laboratorio: mantieni identica una lunga parte statica del system prompt e modifica solo la domanda finale. Confronta TTFT con caching attivo/inattivo, ripeti più volte e misura mediane; un unico esempio è solo indicativo. Verifica l'effettiva opzione con `vllm serve --help | grep -i prefix`.

## 7.3 Bedrock e provider gestiti
Amazon Bedrock supporta, per modelli idonei, **implicit caching** e **explicit caching** tramite checkpoint o controlli di cache specifici della API/modello. Il prefisso statico va collocato prima del contenuto dinamico. La presenza di un checkpoint non garantisce cache hit; verifica i campi di usage della risposta. Limiti minimi di token, TTL, region, pricing e endpoint supportati cambiano in base al modello: consulta sempre la documentazione prima di implementare.

## 7.4 Errori concettuali frequenti
- "La KV cache mi salva i prompt su disco": falso come descrizione generale.
- "Prefix caching significa semantic caching": falso, il primo tipicamente richiede prefissi/token allineati.
- "Il caching risolve le allucinazioni": falso.
- "Un cache hit è sempre economicamente vantaggioso": non necessariamente, dipende da costi di scrittura, lettura, TTL e riutilizzi.

## 7.5 Laboratorio
`laboratori/07_prefix_cache_lab.py` invia più richieste con prefisso fisso e domande variabili, misurando tempi end-to-end. Per una misura vera del TTFT, leggere eventi `stream=True` anziché il tempo totale. Per l'esperimento rigoroso fare due riavvii separati del server, uno con caching attivo e uno disattivato, e verificare metriche del server.

## Autoverifica
1. Qual è la distinzione tecnica tra KV cache e prefix caching?
2. Come massimizzi il riuso di un system prompt condiviso?
3. Perché cambiare la parte iniziale del prompt produce cache miss?
4. Come controlli che Bedrock abbia effettivamente riutilizzato token?
5. Quali problemi crea la semantic cache in un chatbot multi-tenant?

## Videotutorial
https://www.deeplearning.ai/courses/building-toward-computer-use-with-anthropic
https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm
## Riferimenti
https://docs.vllm.ai/en/v0.31.0/features/automatic_prefix_caching/
https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
