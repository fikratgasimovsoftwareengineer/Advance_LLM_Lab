"""Laboratorio didattico: Automatic Prefix Caching con vLLM 0.31.0.

Un solo modulo Python: download del modello, caricamento vLLM, tokenizzazione,
matching a blocchi (simulazione), inferenza e misurazione degli hit reali.

Avvio:
    source ~/myenv/bin/activate
    python vllm_prefix_caching_offline.py

Confronto in un *altro processo*, senza APC:
    python vllm_prefix_caching_offline.py --disable-prefix-cache

IMPORTANTE: PrefixBlockIndex è un modello didattico degli hash; NON calcola e
NON conserva tensori KV. La vera cache KV è gestita internamente da vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field

MODEL_ID = "Qwen/Qwen2.5-0.5B"  # modello pretrained BASE, non Instruct
BLOCK_SIZE = 16                  # token per blocco del laboratorio
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 48
GPU_MEMORY_UTILIZATION = 0.75


def build_document() -> str:
    """Prefisso volutamente lungo e IDENTICO per le richieste A e B."""
    head = (
        "MANUALE OPERATIVO DEL LABORATORIO LLM\n"
        "Il servizio usa un modello linguistico locale con vLLM. "
        "Le sezioni seguenti descrivono un servizio dimostrativo.\n"
    )
    records = "\n".join(
        f"Sezione {i:03d}: il server elabora prompt, usa attenzione causale, "
        "mantiene tensori key e value e restituisce testo."
        for i in range(1, 65)
    )
    return head + records + "\n\n"


def block_hash_chain(token_ids: list[int], size: int = BLOCK_SIZE) -> list[str]:
    """SIMULAZIONE: H_i = SHA256(H_(i-1), token IDs del blocco i).

    Solo i blocchi COMPLETI sono indicizzati. L'hash è didattico: vLLM usa
    una serializzazione propria e può includere chiavi aggiuntive (salt, LoRA).
    """
    parent_hash = "GENESIS"
    hashes: list[str] = []
    n_complete = len(token_ids) // size

    for i in range(n_complete):
        block = token_ids[i * size : (i + 1) * size]
        payload = json.dumps(
            {"parent_hash": parent_hash, "token_ids": block},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        current_hash = hashlib.sha256(payload).hexdigest()
        hashes.append(current_hash)
        parent_hash = current_hash
    return hashes


@dataclass
class PrefixBlockIndex:
    """Indice logico dei blocchi calcolati, NON la cache KV reale di vLLM."""

    size: int = BLOCK_SIZE
    known_hashes: set[str] = field(default_factory=set)

    def predicted_reused_tokens(self, token_ids: list[int]) -> int:
        matching_blocks = 0
        # Il riuso vale soltanto per un PREFISSO contiguo di blocchi già noti.
        for block_hash in block_hash_chain(token_ids, self.size):
            if block_hash not in self.known_hashes:
                break
            matching_blocks += 1
        return matching_blocks * self.size

    def remember(self, token_ids: list[int]) -> None:
        # Didatticamente memorizziamo gli hash, non le matrici K/V.
        self.known_hashes.update(block_hash_chain(token_ids, self.size))


def run_inference(llm, sampler, name: str, token_ids: list[int], index: PrefixBlockIndex):
    """Esegue l'inferenza e confronta matching teorico e cache hit reale."""
    # Calcolo del possibile matching PRIMA dell'inferenza.
    theoretical_hits = index.predicted_reused_tokens(token_ids)
    hashes = block_hash_chain(token_ids)

    begin = time.perf_counter()
    # Passiamo direttamente gli ID dei token: nessuna ambiguità sul prefisso.
    result = llm.generate(
        {"prompt_token_ids": token_ids},
        sampling_params=sampler,
        use_tqdm=False,
    )[0]
    elapsed = time.perf_counter() - begin

    # Informazione misurata dall'ENGINE vLLM, non dalla simulazione.
    real_hits = result.num_cached_tokens

    print(f"\n{'=' * 66}")
    print(f"{name}")
    print(f"Token nel prompt:                 {len(token_ids)}")
    print(f"Blocchi completi nel prompt:    {len(hashes)}")
    print(f"Riuso previsto (solo hash):     {theoretical_hits} token")
    print(f"Riuso effettivo (vLLM):         {real_hits if real_hits is not None else 'non esposto'} token")
    print(f"Tempo totale di generazione:    {elapsed:.3f} s")
    print("Hash dei primi tre blocchi:     ", [h[:12] for h in hashes[:3]])
    print("Risposta generata:")
    print(result.outputs[0].text.strip()[:500])

    index.remember(token_ids)
    return {"case": name, "hits": real_hits, "elapsed": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Laboratorio vLLM APC interamente in Python")
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="disabilita APC per il confronto (avviare in un nuovo processo)",
    )
    args = parser.parse_args()

    # Import delle dipendenze GPU solo quando avviamo il laboratorio.
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    # 1) Download programmato nel modulo Python (riusa la cache HF se presente).
    print(f"[1] Download/verifica modello: {MODEL_ID}")
    local_model_path = snapshot_download(repo_id=MODEL_ID)
    print("    Directory modello:", local_model_path)

    # 2) Caricamento del modello in vLLM e configurazione della cache KV.
    apc_enabled = not args.disable_prefix_cache
    print(f"[2] Caricamento modello su GPU | Prefix caching={apc_enabled}")
    llm = LLM(
        model=local_model_path,
        dtype="auto",
        max_model_len=MAX_MODEL_LEN,
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=apc_enabled,
    )
    tokenizer = llm.get_tokenizer()
    sampler = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    # 3) Tokenizzazione: il prefisso viene tokenizzato UNA volta e riutilizzato.
    shared_prefix_ids = tokenizer.encode(build_document(), add_special_tokens=False)
    a_suffix = tokenizer.encode("Domanda: definisci KV cache.\nRisposta: ", add_special_tokens=False)
    b_suffix = tokenizer.encode("Domanda: cosa significa prefix caching?\nRisposta: ", add_special_tokens=False)
    c_suffix = tokenizer.encode("Domanda: cosa fa vLLM?\nRisposta: ", add_special_tokens=False)
    different_beginning = tokenizer.encode(
        "ARCHIVIO TOTALMENTE DIFFERENTE.\n", add_special_tokens=False
    )

    # A e B hanno l'identico prefisso a livello di TOKEN ID.
    prompt_a = shared_prefix_ids + a_suffix
    prompt_b = shared_prefix_ids + b_suffix
    # C contiene gli stessi token del documento, ma NON più come prefisso.
    prompt_c = different_beginning + shared_prefix_ids + c_suffix

    max_len = max(len(prompt_a), len(prompt_b), len(prompt_c)) + MAX_NEW_TOKENS
    if max_len > MAX_MODEL_LEN:
        raise ValueError(
            f"Prompt troppo lungo: massimo stimato {max_len}, limite {MAX_MODEL_LEN}. "
            "Ridurre il numero di sezioni in build_document()."
        )

    print(f"[3] Token del prefisso condiviso: {len(shared_prefix_ids)}")
    print(f"    Blocchi completi: {len(shared_prefix_ids) // BLOCK_SIZE}")
    print(f"    I primi 8 ID: {shared_prefix_ids[:8]}")
    print("[4] Inferenza sequenziale: stessa istanza del motore e stessa cache GPU")

    index = PrefixBlockIndex(size=BLOCK_SIZE)
    cases = [
        ("A - COLD: prefisso non ancora in cache", prompt_a),
        ("B - WARM: stesso prefisso, domanda nuova", prompt_b),
        ("C - CONTROLLO: inserimento di token all'inizio", prompt_c),
    ]
    report = [run_inference(llm, sampler, title, tokens, index) for title, tokens in cases]

    print(f"\n{'=' * 66}\nRIEPILOGO")
    for row in report:
        print(f"{row['case']}: cached_tokens={row['hits']} | elapsed={row['elapsed']:.3f}s")

    if apc_enabled:
        hits_b = report[1]["hits"]
        if hits_b is not None and hits_b > 0:
            print("\nESITO: prefix caching verificato con cache hit REALI su B.")
        else:
            print("\nESITO: nessun cache hit confermato su B; verificare i log vLLM.")
    else:
        print("\nControllo: APC disabilitato. Gli hash didattici restano calcolabili, ")
        print("ma vLLM non deve riutilizzare blocchi KV attraverso APC.")


if __name__ == "__main__":
    main()
