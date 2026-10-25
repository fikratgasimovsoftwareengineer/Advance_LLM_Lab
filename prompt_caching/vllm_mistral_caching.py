"""
vLLM Inference Demo — Mistral-7B-Instruct-v0.3
Mostra: inferenza base, multi-turn, e Automatic Prefix Caching (APC)

Requisiti:
    pip install vllm>=0.4.0 transformers

GPU consigliata: 16–24 GB VRAM (A10G, RTX 3090/4090, L4)
Per macchine più piccole: usa quantizzazione AWQ/GPTQ (vedi sezione opzionale)
"""

import time
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput


# ─────────────────────────────────────────────
# 1. INIZIALIZZAZIONE DEL MOTORE vLLM
# ─────────────────────────────────────────────

def build_engine(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",   # 3B FP16 ≈ 6 GiB → OK su 8 GiB
    gpu_memory_utilization: float = 0.80,           # lascia ~1.5 GiB per KV cache
    max_model_len: int = 4096,                      # ridotto per risparmiare KV cache
    enable_prefix_caching: bool = True,
    tensor_parallel_size: int = 1,
) -> LLM:
    """
    Inizializza il motore vLLM con APC abilitato.

    Modelli consigliati per GPU ≤ 8 GiB VRAM (FP16, senza quantizzazione):
      - Qwen/Qwen2.5-3B-Instruct      ≈ 6.0 GiB  ← default
      - microsoft/Phi-3.5-mini-instruct ≈ 5.5 GiB
      - google/gemma-2-2b-it           ≈ 4.5 GiB

    Per GPU ≥ 16 GiB passa a:
      - mistralai/Mistral-7B-Instruct-v0.3  ≈ 14 GiB
      - meta-llama/Llama-3.1-8B-Instruct   ≈ 16 GiB

    enable_prefix_caching=True → vLLM costruisce un Radix Tree dei blocchi KV.
    Le richieste con prefisso identico riusano i blocchi già calcolati.
    """
    print(f"[vLLM] Carico modello: {model_id}")
    print(f"[vLLM] Prefix caching: {'ABILITATO' if enable_prefix_caching else 'DISABILITATO'}")

    llm = LLM(
        model=model_id,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        tensor_parallel_size=tensor_parallel_size,
        dtype="float16",           # float16 su GPU consumer (Ampere/Ada); bf16 su A100/H100
        trust_remote_code=True,    # richiesto da Qwen e Phi
    )
    return llm


# ─────────────────────────────────────────────
# 2. HELPER: FORMATTAZIONE CHAT (Mistral instruct)
# ─────────────────────────────────────────────

def format_chatml_prompt(
    system: str,
    history: list[dict],   # [{"role": "user"|"assistant", "content": "..."}]
    user_message: str,
) -> str:
    """
    Formato ChatML — usato da Qwen2.5, Phi-3.5, OpenHermes e molti altri:

        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {user}<|im_end|>
        <|im_start|>assistant

    Il prefix caching è più efficace con ChatML perché il blocco <system>
    è sempre identico e viene cachato dal primo token.
    """
    prompt = ""
    if system:
        prompt += f"<|im_start|>system\n{system}<|im_end|>\n"

    for turn in history:
        role = turn["role"]
        content = turn["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def format_mistral_prompt(
    system: str,
    history: list[dict],
    user_message: str,
) -> str:
    """
    Formato Mistral Instruct v3 (mantenuto per compatibilità con modelli Mistral/AWQ).
    """
    prompt = "<s>"
    for i, turn in enumerate(history):
        if turn["role"] == "user":
            if i == 0 and system:
                prompt += f"[INST] {system}\n\n{turn['content']} [/INST] "
            else:
                prompt += f"[INST] {turn['content']} [/INST] "
        elif turn["role"] == "assistant":
            prompt += f"{turn['content']}</s>"

    if not history and system:
        prompt += f"[INST] {system}\n\n{user_message} [/INST]"
    else:
        prompt += f"[INST] {user_message} [/INST]"
    return prompt


# ─────────────────────────────────────────────
# 3. INFERENZA SINGOLA
# ─────────────────────────────────────────────

def single_inference(llm: LLM, prompt: str, params: SamplingParams) -> str:
    t0 = time.perf_counter()
    outputs: list[RequestOutput] = llm.generate([prompt], params)
    elapsed = time.perf_counter() - t0

    result = outputs[0].outputs[0].text
    tokens_generated = len(outputs[0].outputs[0].token_ids)

    print(f"  [METRICS] Token generati: {tokens_generated} | "
          f"Tempo: {elapsed:.2f}s | "
          f"Throughput: {tokens_generated / elapsed:.1f} tok/s")
    return result


# ─────────────────────────────────────────────
# 4. DEMO — PROMPT CACHING CON SYSTEM PROMPT LUNGO
# ─────────────────────────────────────────────

LONG_SYSTEM_PROMPT = """
Sei un assistente AI specializzato in Machine Learning e LLMOps per un team R&D universitario.
Il tuo compito è rispondere in modo tecnico e preciso su argomenti come:
- Architetture di modelli linguistici (Transformer, Attention, MoE)
- Serving e ottimizzazione di LLM (vLLM, TensorRT-LLM, Ollama)
- RAG (Retrieval-Augmented Generation) e vector databases
- Fine-tuning (LoRA, QLoRA, FLAN, instruction tuning)
- Agenti LLM (LangGraph, Google ADK, AutoGen)
- MLOps e osservabilità (MLflow, LangSmith, Prometheus)
- Sicurezza applicata agli LLM (prompt injection, jailbreak, OWASP LLM Top 10)

Rispondi sempre in italiano a meno che non sia esplicitamente richiesto altro.
Quando fornisci codice, usa Python con type hints e commenti esplicativi.
Struttura le risposte con sezioni chiare.
""".strip()


def demo_prefix_caching(llm: LLM):
    """
    Dimostra il vantaggio del prefix caching:
    - Request 1: prefill completo (system + query1)
    - Request 2: solo query2 viene prefillata; il system prompt è in cache
    """
    params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        repetition_penalty=1.1,
    )

    queries = [
        "Spiega brevemente cos'è il KV cache in un Transformer.",
        "Qual è la differenza tra LoRA e QLoRA?",
        "Cos'è un agente LLM e come si differenzia da un chatbot classico?",
    ]

    print("\n" + "=" * 60)
    print("DEMO: Prefix Caching con System Prompt lungo")
    print("=" * 60)
    print(f"System prompt: {len(LONG_SYSTEM_PROMPT.split())} parole\n")

    history = []

    for i, query in enumerate(queries):
        prompt = format_chatml_prompt(
            system=LONG_SYSTEM_PROMPT,
            history=history,
            user_message=query,
        )

        print(f"\n[Query {i+1}] {query}")
        print(f"  Lunghezza prompt totale: {len(prompt.split())} token approx")
        print(f"  {'→ CACHE HIT atteso sul system prompt' if i > 0 else '→ Prima richiesta — prefill completo'}")

        answer = single_inference(llm, prompt, params)

        print(f"\n  Risposta:\n  {answer[:300]}{'...' if len(answer) > 300 else ''}")

        # Aggiorna la storia per il turno successivo
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


# ─────────────────────────────────────────────
# 5. DEMO — BATCH INFERENCE
# ─────────────────────────────────────────────

def demo_batch_inference(llm: LLM):
    """
    vLLM processa più richieste in parallelo sfruttando
    il continuous batching — molto più efficiente di N chiamate sequenziali.
    """
    params = SamplingParams(temperature=0.5, max_tokens=128)

    batch_queries = [
        format_chatml_prompt(LONG_SYSTEM_PROMPT, [], "Cos'è vLLM?"),
        format_chatml_prompt(LONG_SYSTEM_PROMPT, [], "Elenca 3 vector database per RAG."),
        format_chatml_prompt(LONG_SYSTEM_PROMPT, [], "Cosa significa HITL in un sistema agente?"),
        format_chatml_prompt(LONG_SYSTEM_PROMPT, [], "Spiega il concetto di quantizzazione INT4."),
    ]

    print("\n" + "=" * 60)
    print(f"DEMO: Batch Inference ({len(batch_queries)} richieste parallele)")
    print("=" * 60)

    t0 = time.perf_counter()
    outputs = llm.generate(batch_queries, params)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\n  [METRICS] Batch completato in {elapsed:.2f}s | "
          f"Token totali: {total_tokens} | "
          f"Throughput medio: {total_tokens / elapsed:.1f} tok/s\n")

    for i, output in enumerate(outputs):
        text = output.outputs[0].text.strip()
        print(f"  [R{i+1}] {text[:200]}{'...' if len(text) > 200 else ''}\n")


# ─────────────────────────────────────────────
# 6. OPZIONALE — QUANTIZZAZIONE PER GPU < 16 GB
# ─────────────────────────────────────────────

def build_engine_quantized() -> LLM:
    """
    Alternativa: Mistral-7B quantizzato AWQ INT4 per chi vuole un modello 7B
    su GPU da 8 GiB. Richiede ~4.5 GiB VRAM.

    AWQ (Activation-aware Weight Quantization):
      - Pesi compressi in INT4
      - Attivazioni rimangono FP16 durante l'inferenza
      - Perdita di qualità minima rispetto a FP16 (<1% su benchmark)
    """
    return LLM(
        model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=0.75,   # conservativo: lascia spazio per KV cache
        max_model_len=4096,
        enable_prefix_caching=True,
        trust_remote_code=False,
    )


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Cambio a True per testare la versione quantizzata su GPU piccola
    USE_QUANTIZED = False

    if USE_QUANTIZED:
        engine = build_engine_quantized()
    else:
        engine = build_engine(enable_prefix_caching=True)

    # ── Demo 1: Prefix Caching ──
    demo_prefix_caching(engine)

    # ── Demo 2: Batch Inference ──
    demo_batch_inference(engine)

    print("\n✓ Script completato.")