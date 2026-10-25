# Calendario delle sei settimane

| Settimana | Studio guidato | Laboratorio | Evidenza da consegnare |
|---|---|---|---|
| 1 | Modulo 1 + Modulo 2 (parte I) | API vLLM/Qwen, lettura usage, GPU | Screenshot/log `vllm --version`, `nvidia-smi`, richiesta REST |
| 2 | Modulo 2 (parte II) + Modulo 3 (parte I) | Concorrenza, TTFT, ingestione KB | Tabella latenza e documenti indicizzati |
| 3 | Modulo 3 (parte II) + Modulo 4 (parte I) | Retrieval evaluation, decisione RAG vs FT | HitRate@1/3 e due casi errati annotati |
| 4 | Modulo 4 (parte II) + Modulo 5 (parte I) | LoRA opzionale, primo grafo | Report training o motivazione del mancato training; grafo nodi |
| 5 | Modulo 5 (parte II) + Modulo 6 | Tool contracts, routing, HITL, sicurezza | Ciclo agente limitato e caso approvato/rifiutato |
| 6 | Modulo 7 + Modulo 8 + Modulo 9 | Prefix cache, incidente simulato e capstone | Report comparativo, log diagnostico e presentazione finale |

## Rubrica di verifica individuale (0–4)
0: risposta mancante o sbagliata. 1: definizione vaga, senza terminologia precisa. 2: concetti corretti ma implementazione incompleta. 3: implementazione riproducibile con motivazioni. 4: implementazione documentata, valutata e con limiti/rischi esplicitati.

## Domande trasversali per l'esame
1. Perché una quantizzazione 4-bit non riduce automaticamente tutta la VRAM a un quarto?
2. Che cosa distingue vLLM da una libreria di inferenza Transformers?
3. Se il retrieval ha Recall@5 alto ma risposte scorrette, dove cerchi il problema?
4. Quando useresti fine-tuning invece di RAG? E quando entrambi?
5. Spiega la separazione tra decisione dell'LLM e invocazione effettiva di un tool.
6. Come riprendi un agente dopo un'interruzione e cosa non deve essere ripetuto?
7. Che cosa impone least privilege in un agente con accesso a database e email?
8. In quale situazione un document retrieved diventa vettore d'attacco per prompt injection?
9. Come verifichi un reale cache hit invece di limitarti a misurare un tempo più breve?
10. Quali evidenze porteresti al colloquio per sostenere "ho costruito un RAG valutato"?

## Registro esperimenti
Per ogni laboratorio conserva: data, comando eseguito, versione Python e librerie, modello, sistema operativo, GPU/VRAM, dataset, seed se presente, durata, metriche, screenshot/log minimizzati, osservazioni e limiti. Evita dati reali e credenziali.

## Risposte professionali alle cinque domande iniziali (scheletro)
**LLM installato:** "Ho installato ed eseguito Qwen3-0.6B via vLLM 0.31.0 su Linux x86_64. Ho verificato l'inferenza con API REST; tengo distinti test di laboratorio e produzione."

**Hardware:** "GPU NVIDIA: [modello da `nvidia-smi`]; VRAM: [GB]. Configurazione: [single/multi GPU]." Non riempire i campi senza misurarli.

**LoRA/QLoRA:** "Ho eseguito personalmente [sì/no]; se sì, modello, dataset, framework, hyperparametri, durata, test set e metriche." Non dichiarare esecuzione se hai soltanto letto o preparato lo script.

**RAG:** "Ho costruito [demo/progetto effettivamente usato], con embeddings [x], vector store [y], top-k [z], reranking [se presente], test con [n] domande e metriche [HitRate@k, groundedness, latenza]."

**Serving:** "Ho utilizzato vLLM 0.31.0 con endpoint compatibili OpenAI in locale e ne ho testato le richieste. [Aggiungere benchmark solo dopo averlo eseguito]."
