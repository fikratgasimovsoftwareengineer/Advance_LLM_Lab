# Videolezione 03 — Costruire una Knowledge Base RAG valutabile
**Durata suggerita:** 12-15 minuti. **Riprese:** slide della pipeline, editor Python, terminale, risultati di HitRate@k.

**00:00-02:00 — Il problema.**
«Un assistente che risponde a domande su documenti aggiornati non deve affidarsi soltanto alle conoscenze apprese nel pretraining. RAG aggiunge retrieval prima della generazione. Le fonti rimangono in un archivio modificabile.»

**02:00-05:00 — Ingestione.**
Mostra i sei documenti sintetici in `dati/kb_demo.json`. «Dividiamo un corpus in unità recuperabili, generiamo vettori con un embedding multilingue e li indicizziamo con Chroma. Un database vettoriale non equivale alla KB completa: metadati, permessi, data e versioni contano quanto i vettori.»

**05:00-08:00 — Retrieval e risposta.**
Esegui `python 03_indicizza.py` e `python 03_rag.py "Chi approva le comunicazioni esterne?"`. Indica i tre risultati ordinati per distanza. «Passiamo al modello solo i chunk recuperati, ma sono dati: eventuali istruzioni dentro un PDF non devono diventare comandi dell'agente. Le fonti stampate dal codice non certificano da sole l'accuratezza di ogni frase.»

**08:00-12:00 — Evaluation.**
Esegui `python 03_eval_retrieval.py --k 1` e poi `--k 3`. «HitRate@k misura se il documento atteso è presente tra i candidati. Per valutare l'output, dobbiamo controllare groundedness, completezza, astensione e citazioni. Il piccolo dataset serve per dimostrare il metodo, non per certificare il prodotto.»

**12:00-15:00 — Compito.**
Aggiungi un documento irrilevante e ripeti i test; poi prova una domanda senza fonte. Segna cosa è cambiato e perché. 

**Risorsa:** https://www.deeplearning.ai/courses/building-evaluating-advanced-rag
