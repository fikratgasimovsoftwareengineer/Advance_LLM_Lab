"""Laboratorio 6: Human-in-the-Loop con interrupt/Command.
NON invia davvero messaggi. Demo locale: l'input umano non autentica l'operatore.
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


class ApprovalState(TypedDict, total=False):
    request: str
    draft: str
    decision: str
    sent: bool


def prepare(state: ApprovalState) -> dict:
    return {"draft":f"BOZZA DI COMUNICAZIONE (demo): {state['request']}"}


def review(state: ApprovalState) -> dict:
    # Una pausa persistente avverrebbe qui, in produzione con checkpointer robusto.
    decision = interrupt({"type":"approval_needed", "draft":state["draft"],
                          "allowed_decisions":["approva", "rifiuta"]})
    if decision not in ("approva", "rifiuta"):
        raise ValueError("Decisione non valida")
    return {"decision":decision}


def finish(state: ApprovalState) -> dict:
    if state["decision"] != "approva":
        print("Azioni esterne bloccate: richiesta rifiutata")
        return {"sent":False}
    print("Approvazione acquisita SOLO nel laboratorio: nessun messaggio inviato")
    return {"sent":False}


graph = StateGraph(ApprovalState)
graph.add_node("prepare", prepare)
graph.add_node("review", review)
graph.add_node("finish", finish)
graph.add_edge(START, "prepare")
graph.add_edge("prepare", "review")
graph.add_edge("review", "finish")
graph.add_edge("finish", END)
app = graph.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    config = {"configurable":{"thread_id":"laboratorio-demo-001"}}
    paused = app.invoke({"request":"Annuncio di una nuova lezione nel Campus Demo"}, config=config)
    print("RISULTATO ALLA PAUSA:", paused)
    user_input = input("Decisione [approva/rifiuta]: ").strip().lower()
    if user_input not in ("approva", "rifiuta"):
        raise SystemExit("Solo 'approva' o 'rifiuta' sono ammessi")
    resumed = app.invoke(Command(resume=user_input), config=config)
    print("RISULTATO DOPO RIPRESA:", resumed)
