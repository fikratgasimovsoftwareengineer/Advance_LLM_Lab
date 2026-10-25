from llama_index.llms.vllm import Vllm
from ragas import EvaluationDataset, evaluate
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ragas.metrics import ResponseRelevancy, LLMContextPrecisionWithoutReference
from sentence_transformers import SentenceTransformer

def main():
    # 1) Modello vLLM (max_new_tokens più alto: ragas ha bisogno di risposte JSON lunghe)
    llm = Vllm(
        model="Qwen/Qwen3-8B",
        tensor_parallel_size=4,
        max_new_tokens=512,
        vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
    )
    embed = SentenceTransformer("BAAI/bge-m3")

    # 2) Inferenza di esempio
    domanda = "Qual è la capitale d'Italia?"
    contesto = "Roma è la capitale d'Italia ed è attraversata dal fiume Tevere."
    prompt = f"Contesto: {contesto}\nDomanda: {domanda}\nRispondi in modo breve. /no_think"
    risposta = llm.complete(prompt).text.strip()
    print("Risposta:", risposta)

    # 3) Valutazione con ragas
    dataset = EvaluationDataset.from_list([{
        "user_input": domanda,
        "retrieved_contexts": [contesto],
        "response": risposta,
    }])
    risultato = evaluate(
        dataset,
        metrics=[ResponseRelevancy(strictness=1), LLMContextPrecisionWithoutReference()],
        llm=LlamaIndexLLMWrapper(llm),
        embeddings=LlamaIndexEmbeddingsWrapper(embed),
    )
    print(risultato)


if __name__ == "__main__":
    main()