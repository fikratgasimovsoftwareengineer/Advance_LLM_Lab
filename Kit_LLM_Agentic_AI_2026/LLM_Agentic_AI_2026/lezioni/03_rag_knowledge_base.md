# Modulo 3 — Retrieval-Augmented Generation e Knowledge Base
**Durata:** 5 ore.

## 3.1 Che problema risolve
Un modello generativo può non conoscere i dettagli aggiornati di un regolamento o di un catalogo. Il RAG recupera documenti esterni pertinenti alla domanda e li inserisce nel contesto della generazione. La fonte *resta esterna* ai pesi: si può aggiornare la KB senza riaddestrare l'LLM. Ciò non garantisce automaticamente veridicità: recupero sbagliato, citazioni inventate, documenti obsoleti o istruzioni malevole possono compromettere le risposte.

## 3.2 Pipeline di ingestione
`documenti → estrazione del testo → pulizia/versionamento → chunking → embeddings → indicizzazione`. Ogni chunk dovrebbe conservare almeno `document_id`, URL/percorso, titolo, data, versione, permessi e intervallo/posizione d'origine. Il chunking a caratteri è semplice ma non allineato ai token: in produzione provare segmentazione semantica o per sezione, overlap controllato e misure di recall.

Gli **embedding** trasformano testi o domande in vettori. La similarità coseno misura la vicinanza angolare tra vettori normalizzati. Il modello di embedding può essere diverso dall'LLM generativo; lo deve essere quasi sempre, in un RAG classico. Per contenuti multilingue, scegliere un embedding realmente multilingue e valutarlo nella lingua degli utenti.

Il **vector DB** memorizza vettori, metadati e indici efficienti. Esempi: Chroma (locale), Qdrant, FAISS (libreria ANN, non un DB completo), pgvector. La ricerca può essere densa (semantic embeddings), lessicale (BM25) o **ibrida** con fusione dei risultati. Un **reranker** riordina i documenti candidati con una stima più costosa ma potenzialmente più accurata.

## 3.3 Query time
`domanda → embedding della domanda → retrieve top-k → (eventuale BM25+fusion) → rerank → contesto con source IDs → prompt → risposta → citazioni`. Il prompt va scritto per utilizzare i documenti come **dati non attendibili**: un documento non può ridefinire il ruolo dell'assistente o autorizzare tool.

Controllare la lunghezza dei chunk selezionati: troppo testo peggiora costo/latency e può diluire le evidenze. Non confondere rilevanza dei documenti con fedeltà della risposta. Un buon retrieval può essere vanificato da una generazione sbagliata.

## 3.4 Evaluation: una condizione per parlare di produzione
Separare due categorie:
- **Retrieval**: Recall@k/HitRate@k, MRR, nDCG, copertura di domande e permessi.
- **Generazione**: faithfulness/groundedness, correctness rispetto a gold answers, completezza, qualità delle citazioni, percentuale di astensione quando le fonti sono insufficienti, latenza e costo.

Creare almeno 20-50 casi di test con: domanda, risposta attesa, source ID atteso, difficoltà, categoria, lingua e comportamento richiesto in assenza di fonti. Tenere un insieme separato per modifiche e regressioni. Una valutazione automatica con un altro LLM è utile ma non sostituisce controlli umani e criteri espliciti.

## 3.5 Laboratorio
Il dataset `dati/kb_demo.json` è **fittizio**, non contiene regolamenti reali. I programmi `03_indicizza.py` e `03_rag.py` creano una KB Chroma persistente, usano un modello sentence-transformers multilingue e interrogano l'LLM locale tramite API vLLM. `03_eval_retrieval.py` calcola HitRate@k sul dataset campione. Dopo il baseline: prova chunk diversi, top-k differenti, documenti non pertinenti e una domanda senza fonte, monitorando le variazioni.

## Autoverifica
1. Qual è la differenza tra indice vettoriale e KB completa?
2. Perché un reranker non sostituisce l'embedding model?
3. Cosa misura Recall@5?
4. Perché non basta testare soltanto la qualità della risposta finale?
5. Come gestisci cancellazione/versione/permessi dei documenti?

## Video suggerito
https://www.deeplearning.ai/courses/building-evaluating-advanced-rag
