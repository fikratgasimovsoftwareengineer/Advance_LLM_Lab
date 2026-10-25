# Modulo 1 — Fondamenti degli LLM
**Durata:** 3 ore. **Prerequisito:** Python base.

## Obiettivi
Saper spiegare la generazione autoregressiva, i componenti del Transformer, i token, le finestre di contesto, il campionamento e le differenze fra pretraining, instruction tuning e inferenza.

## 1.1 Che cosa apprende un LLM
Un LLM autoregressivo riceve una sequenza di token e stima una distribuzione di probabilità per il token successivo: `P(token_t | token_1, ..., token_(t-1))`. La generazione ripete questo processo. Il modello **non** è un database fattuale: comprime regolarità statistiche e relazioni apprese in parametri. Può produrre affermazioni fluenti ma false, soprattutto su fatti assenti, mutati nel tempo o ambigui.

Il *tokenizer* trasforma il testo in identificatori numerici. I token non coincidono necessariamente con parole: una parola può contenere più token. Il costo dell'inferenza dipende dai token effettivi, non dai caratteri. `max_tokens` limita l'output e può troncarlo anche quando la risposta sarebbe semanticamente incompleta.

## 1.2 Dal testo alla previsione
Il flusso essenziale è: tokenizer → embedding di token → blocchi Transformer (self-attention + MLP con normalizzazione e connessioni residue) → proiezione sul vocabolario → logits → scelta del token. La *causal mask* impedisce a una posizione di leggere i token futuri durante l'inferenza autoregressiva.

La self-attention calcola, in forma semplificata, `Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V`. In un Transformer decoder, i vettori Q (query), K (key), V (value) derivano dalle rappresentazioni nascoste. Con più teste, il modello può apprendere relazioni differenti. Un numero elevato di parametri indica capacità, non garanzia di affidabilità.

## 1.3 Decodifica e parametri
`temperature=0` tende a privilegiare i token più probabili; valori maggiori favoriscono varietà. `top_p` limita il campionamento al sottoinsieme cumulativo di probabilità; `top_k`, ove disponibile, limita il numero di candidati. `max_tokens` è un limite di output. Il risultato può cambiare per versioni del modello, hardware, scheduler e determinismo numerico: la temperatura non è da sola una garanzia di riproducibilità.

Tre fasi da distinguere: **pretraining** (apprendimento generale), **fine-tuning/alignment** (adattamento del comportamento) e **inference** (uso dei pesi per rispondere). La modifica di un prompt non modifica i pesi del modello.

## 1.4 La risposta del nostro Qwen3
Nel test reale si vedeva `prompt_tokens=42`, `completion_tokens=256`, `finish_reason="length"`. È una risposta valida a livello API ma troncata; non prova che il modello abbia concluso. Si vedeva anche `<think> ... </think>` nel `content`: il comportamento è legato alla modalità di reasoning/template del modello e va gestito nel client, non scambiato per una seconda richiesta. Un modello piccolo può avere output in italiano poco corretti, anche se il serving funziona.

## Esperimento
Lancia `python laboratori/01_inferenza_client.py`; confronta `max_tokens=64` e `max_tokens=400`; annota token input/output, `finish_reason`, durata, risposta e presenza o meno dei marcatori `<think>`.

## Autoverifica
1. Perché 100 caratteri non sono sempre 100 token?
2. Che differenza esiste tra logits e probabilità?
3. Qual è il ruolo della causal mask?
4. Perché `finish_reason=length` non indica un errore dell'API?
5. Cosa cambia, e cosa **non** cambia, se aumenti `temperature`?

## Video suggerito
https://www.deeplearning.ai/courses/how-transformer-llms-work
