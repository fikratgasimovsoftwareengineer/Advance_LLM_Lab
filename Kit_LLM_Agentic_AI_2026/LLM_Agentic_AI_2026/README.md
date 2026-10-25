# Fondamenti LLM e Agentic AI — kit di apprendimento (ottobre 2026)

**Lingua:** italiano. **Livello:** intermedio → avanzato. **Durata orientativa:** 36 ore in 6 settimane.

Il pacchetto contiene materiali ORIGINALI: nove lezioni teoriche, nove copioni per videolezioni, esempi Python da eseguire sul proprio computer, dati di esempio inventati e schede per la valutazione. I link ai video esterni rimandano alle rispettive piattaforme: **non sono video prodotti o scaricati nel kit**. L'esecuzione dei laboratori richiede dipendenze e, ove indicato, accesso a Internet per scaricare i modelli.

## Punto di partenza
- Linux x86_64, vLLM `0.31.0`, `Qwen/Qwen3-0.6B` già avviato con successo via API OpenAI-compatible.
- Il laboratorio parte dall'endpoint `http://127.0.0.1:8000/v1`.
- GPU e VRAM **non risultano ancora documentate**: verificare con `nvidia-smi`. Non supponiamo né VRAM né capacità di eseguire fine-tuning.
- Prima di un fine-tuning, fermare il processo `vllm serve` se usa la stessa GPU.

## Struttura
| Modulo | Durata | Teoria | Laboratorio | Video |
|---|---:|---|---|---|
| 01 Fondamenti LLM | 3 h | `lezioni/01_...` | `laboratori/01_inferenza_client.py` | `videolezioni/01_...` |
| 02 Local LLM & Serving | 4 h | `lezioni/02_...` | `laboratori/02_benchmark_concorrenza.py` | `videolezioni/02_...` |
| 03 RAG & Knowledge Base | 5 h | `lezioni/03_...` | `laboratori/03_indicizza.py`, `03_rag.py`, `03_eval_retrieval.py` | `videolezioni/03_...` |
| 04 Fine-Tuning | 5 h | `lezioni/04_...` | `laboratori/04_finetuning_lora.py` | `videolezioni/04_...` |
| 05 Agents & Orchestration | 6 h | `lezioni/05_...` | `laboratori/05_agent_langgraph.py` | `videolezioni/05_...` |
| 06 Platform, Security & HITL | 4 h | `lezioni/06_...` | `laboratori/06_hitl.py` | `videolezioni/06_...` |
| 07 Prompt Caching | 3 h | `lezioni/07_...` | `laboratori/07_prefix_cache_lab.py` | `videolezioni/07_...` |
| 08 Troubleshooting | 3 h | `lezioni/08_...` | `laboratori/08_diagnostica.py` | `videolezioni/08_...` |
| 09 Progetto finale | 3 h | `lezioni/09_...` | integrazione dei laboratori precedenti | `videolezioni/09_...` |

**Sequenza raccomandata:** leggi la lezione, guarda il video esterno, segui il copione della videolezione per riprodurre i passaggi, esegui il codice, rispondi alle domande e confronta le metriche misurate.

## Installazione per i laboratori
Il server vLLM può restare attivo nel suo ambiente `myenv`. Utilizzare un **altro** ambiente isolato per gli strumenti applicativi, evitando conflitti con PyTorch e vLLM:

```bash
cd ~/vllm-lab
python3 -m venv .venv-corso
source .venv-corso/bin/activate
python -m pip install --upgrade pip
pip install -r /percorso/LLM_Agentic_AI_2026/laboratori/requirements-app.txt
```

Per il fine-tuning, usare un terzo ambiente Python e `requirements-train.txt`; richiede GPU compatibile e memoria sufficiente. Le dipendenze possono modificarsi: preferire un lockfile dopo il primo test riuscito.

## Collegamenti alle videolezioni esterne
Lezioni e riferimenti completi: `risorse/fonti_e_video.md`.

## Criteri di valutazione
Un esercizio è completato soltanto quando esiste una **prova riproducibile**: codice, comandi, output, metriche, limitazioni e spiegazione delle scelte. Eseguire un esempio non equivale ad avere esperienza di produzione.
