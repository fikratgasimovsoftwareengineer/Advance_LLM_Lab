# Modulo 5 — Agentic AI, tool calling, ReAct, LangGraph e MCP
**Durata:** 6 ore.

## 5.1 Definizioni da non confondere
Un **workflow** è una sequenza di operazioni con regole definite dal programma. Un **agente** delega alcune decisioni al modello, per esempio quale strumento chiamare o quale sotto-obiettivo perseguire. Un **tool** è una funzione esterna (ricerca, interrogazione SQL, RAG, calcolo, API). Il tool calling richiede *due passaggi*: il modello propone una chiamata strutturata e il codice applicativo la valida/esegue; il modello non possiede automaticamente permessi di sistema.

Il paradigma **ReAct** alterna elaborazione del problema, selezione di azioni, osservazione dei risultati e iterazione. Un **planner-executor** separa la pianificazione dall'esecuzione; un **router** classifica la richiesta e seleziona un ramo; un **critic/editor** controlla la bozza e decide se correggerla. La presenza di più nodi non garantisce un miglioramento: aumenta latenza, costo e punti di errore. Limitare passi e cicli.

## 5.2 Stato e grafo
In LangGraph i nodi sono funzioni che ricevono uno stato e restituiscono aggiornamenti; gli edge definiscono i percorsi, anche condizionali. Lo **State** descrive i dati condivisi. Un *checkpointer* salva lo stato per ripresa, HITL e debugging; uno store persistente può ospitare memoria a lungo termine. Memoria conversazionale, cache e KB hanno scopi differenti.

Per una pipeline di analisi di mercato: `intent_router → researcher → analyzer → writer → critic → editor → final_answer`. Per ogni nodo specificare **input, trasformazione, output, errori**. Il router può classificare `technology`, `economics`, `jobs`, `sport`, `scientific_research`; il researcher può usare Tavily o una KB. Le dipendenze iniettate (`IntentRouterNode(OllamaRouter(settings))`) sono una **relazione di composizione/delega**, non un edge di esecuzione del grafo: UML delle classi e grafo dati vanno rappresentati separatamente.

## 5.3 Tool contracts e structured outputs
Definire schema dei parametri, campi obbligatori, limiti temporali, whitelist di domini/risorse, ritentativi idempotenti e forma dei risultati. Un tool dovrebbe restituire errori leggibili dal codice, non contenuto HTML non filtrato. Quando il modello produce JSON, validarlo con `pydantic`/JSON schema. Evitare di fidarsi di un campo `approved` inventato dal modello.

## 5.4 MCP (Model Context Protocol)
MCP è uno standard che permette a client e server di esporre strumenti e risorse attraverso un'interfaccia comune. Non equivale a un framework multi-agent e non sostituisce autorizzazione, autenticazione o sandbox. Occorre trattare tool description e risposte esterne come superfici non attendibili, stabilire permessi granulari e controllare le operazioni con effetti esterni.

## 5.5 Agentic RAG
In un RAG classico, il retriever spesso viene chiamato per ogni domanda. In un **Agentic RAG**, l'agente può scegliere se interrogare una KB, fare un'altra ricerca, chiedere chiarimenti o confrontare fonti, ma questa autonomia deve essere misurata. Valutare: tool selection accuracy, task success, numero di step, qualità delle prove, costo e fallimenti per tipo.

## 5.6 Laboratorio
`laboratori/05_agent_langgraph.py` realizza un piccolo grafo locale con tre funzioni: ricerca su dati fittizi, stesura tramite vLLM e verifica deterministica delle fonti. Espandilo progressivamente fino alla pipeline router/research/analyzer/writer/critic/editor. Non iniziare da sette agenti indipendenti: prima dimostrare che una singola pipeline a tre nodi produca risposte controllabili.

## Autoverifica
1. Qual è la differenza tra tool calling e accesso effettivo a uno strumento?
2. In LangGraph che cosa distingue un nodo da un edge?
3. Qual è il ruolo dello stato e di un checkpointer?
4. In cosa differiscono ReAct, router e planner-executor?
5. Come eviti cicli illimitati, tool ripetuti e costi imprevedibili?

## Videotutorial
https://academy.langchain.com/courses/intro-to-langgraph
https://www.deeplearning.ai/courses/ai-agents-in-langgraph
https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/agentic-rag
