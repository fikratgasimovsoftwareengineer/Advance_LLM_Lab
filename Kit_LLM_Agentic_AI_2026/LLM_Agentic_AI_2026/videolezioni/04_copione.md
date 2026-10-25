# Videolezione 04 — Fine-tuning con LoRA e QLoRA
**Durata suggerita:** 12-15 minuti. **Riprese:** diagramma dei pesi congelati/adapters, editor e training log se l'hardware lo consente.

**00:00-03:00 — La scelta progettuale.**
«Prima di addestrare un LLM domandiamoci se vogliamo cambiare le fonti informative oppure il comportamento. Per conoscenza aggiornata e citabile, spesso vince RAG. Se il modello sbaglia sistematicamente un formato o un task, può essere utile il fine-tuning.»

**03:00-06:00 — Tre tecniche.**
Mostra W, A e B su slide. «Full fine-tuning aggiorna i pesi del modello. LoRA impara una correzione di rango ridotto mantenendo fissi i pesi base. QLoRA carica il modello base quantizzato, riducendo la memoria per gli stessi adapter. Più piccolo non equivale a gratuito: servono RAM, attivazioni, ottimizzatore e verifiche.»

**06:00-09:00 — Dataset.**
Apri `04_finetuning_lora.py`: «Qui ci sono esempi sintetici di domande e risposte. Lo scopo è imparare la struttura del dataset, la chat template, il rank `r` e i target modules. Dieci esempi di train e due di validazione non misurano un miglioramento reale.»

**09:00-12:00 — Laboratorio opzionale sulla GPU.**
Prima di eseguire: mostra `nvidia-smi`, spegni vLLM e attiva ambiente training. Esegui solo se CUDA e VRAM sono compatibili. Osserva loss e adapter salvato, evitando conclusioni assolute. Se la macchina non è adeguata, presenta il codice e i prerequisiti senza simulare un training eseguito.

**12:00-15:00 — Autoverifica.**
«Ho davvero aggiornato pesi o soltanto cambiato un prompt? Posso dimostrare un miglioramento su domande non viste? Che cosa devo dichiarare a un colloquio tecnico?».

**Risorsa:** https://www.deeplearning.ai/courses/finetuning-large-language-models
