# Note importanti sull'esecuzione e la verifica

- Gli script del kit sono **esempi didattici originali**, controllati per la sintassi Python e per la coerenza dei file locali.
- Non sono stati lanciati sul Dell G16 dell'utente: GPU/VRAM, driver, dipendenze, versioni di LangGraph/TRL/Chroma e tempi reali devono essere verificati nell'ambiente di destinazione.
- Usare il proprio ambiente `myenv` per il server vLLM e un altro ambiente per gli script client/RAG; fermare vLLM prima di un esperimento di training sulla stessa GPU.
- Il dataset Campus Demo è inventato, con URL su example.org (risorsa segnaposto). Non indica condizioni o regolamenti di un'università reale.
- Uno script dimostrativo RAG non equivale a un sistema di produzione: servono controllo accessi ai documenti, pipeline di aggiornamento/cancellazione, test, monitoring, hardening, backup e governance.
- I file `videolezioni/*.md` sono **copioni di video**: non sono riprese né tracce audio registrate. I video già pubblicati sono raggiungibili tramite i link presenti in `risorse/fonti_e_video.md`.
