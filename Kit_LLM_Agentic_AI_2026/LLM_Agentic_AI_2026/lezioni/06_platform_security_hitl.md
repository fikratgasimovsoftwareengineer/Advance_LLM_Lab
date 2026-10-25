# Modulo 6 — Platform, Security e Human-in-the-Loop
**Durata:** 4 ore.

## 6.1 Piattaforma: dove si trova ogni componente
Separare **control plane** (identità, configurazioni, orchestrazione), **data plane** (inferenza, retrieval, tool execution) e **observability plane** (log, metriche, trace). In locale, vLLM serve il modello e il backend Python gestisce retrieval e agenti. In cloud, si possono utilizzare Bedrock per modelli gestiti o GPU self-hosted su EC2/EKS/ECS EC2; evitare l'errore concettuale per cui ECR sarebbe un orchestratore: ECR conserva immagini, ECS/EKS gestiscono esecuzione, IAM gestisce autorizzazioni.

Un percorso di deploy tipico per modelli self-hosted: Git → CI/test → image build → ECR → ECS su EC2 o EKS con nodi GPU → ALB/ingress → API. Bedrock sostituisce il componente model serving ma non tutta l'applicazione. Specificare sempre region, model access, IAM role/STS identity, secret management, timeouts e retry policy.

## 6.2 Fiducia tra componenti
Il **prompt injection** sfrutta istruzioni malevole in input non attendibili: email, pagine web, documenti recuperati dal RAG, output di un tool. La difesa non è semplicemente "dire al modello di ignorare". Serve una barriera **nel codice**: classificazione dei dati, access control verificato sul server, validazione dei parametri, permission boundary per tenant, URL/path allowlist, tool limitati in scrittura, output escaping, auditing e test avversariali.

Il modello non deve poter invocare direttamente comandi shell arbitrari, cancellare risorse o mandare email senza controllo. Non includere chiavi API, dati privati o credenziali in prompt, log o dataset. I documenti della KB possono essere attendibili come fonte di fatti ma **non** come fonte di istruzioni per l'agente.

## 6.3 Human-in-the-Loop (HITL)
HITL significa sospendere l'esecuzione **prima** di un'azione sensibile: il sistema crea una richiesta di revisione, salva lo stato, espone all'operatore il piano e attende una decisione. Esiti tipici: approva, modifica, rifiuta. Una richiesta approvata può riprendere **solo** se l'identità e i permessi dell'approvatore sono stati verificati sul backend.

In LangGraph, `interrupt(...)` consente una pausa nel grafo e `Command(resume=...)` la ripresa. La persistenza non è opzionale in uno scenario reale: un checkpointer in memoria serve solo per demo e non resiste ai riavvii. Attenzione agli effetti collaterali eseguiti **prima** dell'interrupt: il nodo potrebbe essere rieseguito dopo la ripresa. Usare azioni idempotenti e spostare gli effetti dopo l'approvazione.

## 6.4 Esercizio pratico
`laboratori/06_hitl.py` realizza una revisione umana su una bozza di messaggio, senza inviarlo realmente. Esegui il caso approvato, poi quello rifiutato. Spiega perché l'approvazione simulata da input locale non è sufficiente come autenticazione in un servizio multi-utente.

## Autoverifica
1. Cosa succede se un PDF indicizzato scrive "ignora tutte le regole"?
2. Perché un modello non può approvare da solo una sua operazione sensibile?
3. Come distingui IAM role, credenziali AWS e accesso al modello Bedrock?
4. Perché un checkpointer persistente serve per HITL robusto?
5. Dove metti i controlli di autorizzazione per uno strumento di cancellazione?

## Videotutorial
https://www.deeplearning.ai/courses/safe-and-reliable-ai-via-guardrails
https://academy.langchain.com/courses/intro-to-langgraph
## Riferimento di sicurezza
https://genai.owasp.org/llmrisk/llm01-prompt-injection/
