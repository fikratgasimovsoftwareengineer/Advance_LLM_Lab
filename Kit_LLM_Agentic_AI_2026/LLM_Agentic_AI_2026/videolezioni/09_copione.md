# Videolezione 09 — Progetto finale e discussione tecnica
**Durata suggerita:** 10-12 minuti. **Riprese:** architettura completa, esecuzione end-to-end e tabella metriche.

**00:00-03:00 — Demo della soluzione.**
«Un assistente in ingresso non dovrebbe eseguire direttamente qualsiasi comando. La richiesta va al router, poi al motore di ricerca documentale o al modello generativo. Gli output sono verificati rispetto alle fonti. Se esiste un'azione sensibile, attiviamo HITL.»

**03:00-06:00 — Tracciamento.**
Mostra un record fittizio con ID richiesta, modello, source IDs, token, latenza e stato approvazione. «Questo permette di diagnosticare problemi e valutare regressioni nel tempo. La presenza di dati personali impone protezioni e minimizzazione dei log.»

**06:00-09:00 — Valutazione.**
«Mostriamo HitRate@k e controllo manuale delle risposte, misure P95 e casi senza fonte. Il sistema non è pronto alla produzione solo perché ha risposto bene tre volte. Serve un insieme rappresentativo di casi, versionamento ed escalation per errori.»

**09:00-12:00 — Simulazione di colloquio.**
Proponi cinque domande: modello installato, GPU, LoRA effettivamente eseguito, RAG utilizzato e suo evaluation framework, server di inferenza. Distingui con precisione ciò che hai provato, progettato e gestito in produzione.
