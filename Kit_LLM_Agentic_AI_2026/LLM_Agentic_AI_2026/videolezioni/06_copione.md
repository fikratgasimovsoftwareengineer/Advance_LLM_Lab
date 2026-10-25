# Videolezione 06 — Security e Human-in-the-Loop
**Durata suggerita:** 10-13 minuti. **Riprese:** confini di fiducia, demo `interrupt`, diagramma IAM/RBAC.

**00:00-03:00 — Documenti non attendibili.**
«Un agente RAG può recuperare il testo di una pagina che contiene 'ignora il sistema ed esegui questo comando'. Per il modello è testo, ma nel progetto non deve diventare autorità. Isoliamo istruzioni del sistema da dati non attendibili e imponiamo permessi nel backend.»

**03:00-05:00 — Difesa a strati.**
Mostra: input validation → auth → tool policy → approval → audit. «Un guardrail testuale può fallire. Non affidiamo all'LLM la decisione finale di leggere dati di altri utenti o modificare risorse.»

**05:00-09:00 — Human-in-the-Loop.**
Apri `06_hitl.py`. «L'interrupt sospende il grafo prima dell'azione. Un approvatore umano può scegliere, e il programma riprende con `Command(resume=...)`. Nel demo l'input è locale, non autenticato, quindi non è ancora un sistema di approvazione enterprise.»

**09:00-13:00 — Platform.**
Disegna Git/CI, ECR, EKS/ECS su EC2, IAM, vLLM/Bedrock. «ECR conserva immagini, non le avvia. IAM identifica e autorizza. In Bedrock controlliamo credenziali, region e accesso al modello separatamente. Il principio del minimo privilegio vale per agenti e applicazioni.»

**Risorsa:** https://www.deeplearning.ai/courses/safe-and-reliable-ai-via-guardrails
