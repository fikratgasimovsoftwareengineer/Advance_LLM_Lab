# Modulo 9 — Progetto conclusivo: RAG agentico verificabile
**Durata:** 3 ore per l'integrazione base; estensioni facoltative 8-12 ore.

## Scenario
Progettare un assistente che risponde a domande su una knowledge base dimostrativa, distingue quesiti documentali da domande generali, cita le fonti, registra i risultati e sottopone alla revisione umana ogni azione sensibile. L'esempio deve usare solo i documenti **sintetici** in `dati/`, non dati reali di studenti.

## Architettura richiesta
`Utente → API backend → Intent Router → [KB Retriever / General LLM] → Risposta con fonti → Critic/Eval → Output`. Se la richiesta necessita una modifica o un invio verso sistemi esterni, instradarla a `HITL approval → tool con autorizzazione verificata → audit`. Il sistema LLM locale resta un backend di inferenza, non decide da solo l'accesso alle informazioni o ai tool.

## Requisiti minimi di consegna
1. Un endpoint Python o script CLI interrogabile che usa il modello Qwen in vLLM.
2. Un archivio di documenti indicizzato con un embedding model differente dal modello generativo.
3. Retrieval top-k con ID delle fonti e comportamento di astensione quando la documentazione non basta.
4. Un workflow LangGraph con stato esplicito, un ramo condizionale e un limite massimo di ripetizioni.
5. Un esempio HITL di approvazione/rifiuto senza eseguire azioni irreversibili.
6. Un set di almeno 20 domande di test; 5 per ciascuna categoria: semplici, multi-hop, senza risposta, tentativi di prompt injection.
7. Report con HitRate@k, almeno una misura di groundedness giudicata manualmente, latenza mediana/P95, error rate e limite di memoria usata.

## Criteri di valutazione (100 punti)
| Dimensione | Peso |
|---|---:|
| Correttezza tecnica e riproducibilità | 20 |
| Retrieval ed evidence/citazioni | 20 |
| Esecuzione agentica controllata | 15 |
| Fine-tuning: progettazione e motivazione (anche senza GPU adeguata) | 10 |
| Sicurezza, autorizzazioni e HITL | 15 |
| Tracing, metriche, test e rollback | 15 |
| Documentazione delle decisioni e limitazioni | 5 |

## Discussione orale finale
Preparare risposte sintetiche a cinque domande: quale modello open-weight hai eseguito? Su quale GPU/VRAM? Hai svolto LoRA/QLoRA personalmente? Hai realizzato e valutato un RAG realmente utilizzato? Quale inference server hai configurato? Distinguere esperienza laboratoriale, sperimentale e produzione reale: non attribuire a un laboratorio didattico risultati di produzione.

## Direzioni di approfondimento
Multi-agent evaluation; router adattivi; speculative decoding; agent memory; guardrail e test red-team; Model Context Protocol e autorizzazione dei tool; A/B testing; deployment EKS GPU; telemetry OpenTelemetry; semantic cache con invalidazione e isolamento per tenant.
