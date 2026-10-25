# Modulo 4 — Fine-Tuning, LoRA e QLoRA
**Durata:** 5 ore.

## 4.1 Quando scegliere il fine-tuning
Il fine-tuning adatta i parametri (o una loro piccola componente aggiuntiva) a esempi supervisionati. È indicato quando bisogna rendere stabile uno **stile**, un **formato strutturato**, una **tassonomia**, un comportamento di tool calling, o migliorare su un compito ricorrente. RAG è spesso preferibile per fonti aggiornabili, verificabili e citabili. I due approcci possono convivere: fine-tuning per comportamento, RAG per conoscenza accessibile e aggiornata.

Non usare fine-tuning per correggere un indice RAG difettoso o per mettere nel modello segreti e dati personali non autorizzati. Prima di addestrare, confronta un *baseline* che usa buon prompting e RAG.

## 4.2 Tecniche
**Full fine-tuning** modifica tutti i pesi e può richiedere hardware molto costoso. **LoRA** congela i pesi base e apprende matrici a basso rango che approssimano l'adattamento di alcune trasformazioni lineari: `W' = W + ΔW`, con `ΔW ≈ B A`. Parametri importanti: `r`, `lora_alpha`, `lora_dropout`, `target_modules`, learning rate, steps/epochs. Con **QLoRA**, il modello base è caricato in forma quantizzata, tipicamente a 4 bit, mentre gli adapter sono allenati con precisione appropriata: ciò riduce memoria, ma non annulla i costi di attivazioni e optimizer.

**SFT** usa esempi input → output e una loss di previsione token; DPO e tecniche di preference optimization servono ad altri obiettivi. Non confondere un adapter LoRA con un intero modello indipendente: per inferenza serve base model + adapter, salvo fusione compatibile.

## 4.3 Dataset e qualità
Ogni esempio deve rappresentare il task desiderato, essere revisionato, non includere risposte contraddittorie o PII non autorizzati e rispettare la chat template del modello. Separare train/dev/test **prima** di costruire prompt o riutilizzare esempi. Un microdataset sintetico serve per verificare la pipeline, non per dimostrare un aumento di qualità.

Misure prima/dopo: exact match o F1 dove appropriato, validità JSON, accuratezza della classificazione, allucinazioni, safety regressions, qualità della lingua, costo inferenziale. Annotare seed, checkpoint, versione librerie, dataset, hyperparameters e benchmark riproducibili.

## 4.4 Laboratorio
`laboratori/04_finetuning_lora.py` propone una sessione **dimostrativa** con Qwen3-0.6B, TRL e PEFT, dataset fittizio e LoRA. L'esecuzione richiede GPU adeguata e un ambiente training isolato. Prima interrompere vLLM, che può già occupare molta VRAM. Salvare gli adapter; poi confrontare alcuni prompt non visti con il modello di partenza. Non considerare un training terminato con successo come prova di un miglioramento senza evaluation indipendente.

## Autoverifica
1. Cosa viene aggiornato in LoRA?
2. In QLoRA che cosa è quantizzato?
3. Quando il RAG è più corretto rispetto al fine-tuning?
4. Cosa misura un test set separato?
5. Perché un dataset di 12 esempi non dimostra una competenza di produzione?

## Video suggerito
https://www.deeplearning.ai/courses/finetuning-large-language-models
## Riferimento tecnico
https://huggingface.co/docs/trl/peft_integration
