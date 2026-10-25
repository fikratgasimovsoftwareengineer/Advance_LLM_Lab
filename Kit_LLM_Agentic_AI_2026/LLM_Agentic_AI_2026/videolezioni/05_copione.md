# Videolezione 05 — Dal modello all'agente LangGraph
**Durata suggerita:** 14-18 minuti. **Riprese:** diagramma di stato, codice del grafo e trace dei nodi.

**00:00-03:00 — Agente vs workflow.**
«Un workflow chiama funzioni secondo il programma; un agente lascia alcune decisioni al modello. In entrambi i casi, i permessi appartengono al software, non alla frase prodotta dall'LLM. Tool calling è una richiesta strutturata, eseguita solo dopo validazione.»

**03:00-06:00 — Il grafo.**
Disegna `retrieve → draft → verify → {done/retry}`. «Ogni nodo legge lo stato e ne restituisce aggiornamenti. Un edge trasferisce il controllo, uno condizionale decide il ramo. Un limite di tentativi evita il ciclo infinito.»

**06:00-10:00 — ReAct e ruoli.**
«ReAct usa osservazioni dei tool per scegliere l'azione successiva. Router, planner, researcher, critic ed editor sono ruoli che possiamo implementare in un grafo: non è obbligatorio usare un LLM diverso per ciascuno. Introduciamo un agente solo quando il comportamento dinamico porta valore misurabile.»

**10:00-15:00 — Esecuzione.**
Mostra `05_agent_langgraph.py` e stampa stato finale, fonti e `attempt`. Spiega che il controllo `[Dxx]` non certifica groundedness. Proponi un diagramma delle classi separato da un grafico dei nodi: una dipendenza Python verso un oggetto router non è un edge del workflow.

**15:00-18:00 — Verifica.**
«Che cosa persiste tra due chiamate? Quali strumenti può usare il nodo? Che cosa accade se il modello produce una citazione inventata?».

**Risorse:** https://academy.langchain.com/courses/intro-to-langgraph — https://www.deeplearning.ai/courses/ai-agents-in-langgraph
