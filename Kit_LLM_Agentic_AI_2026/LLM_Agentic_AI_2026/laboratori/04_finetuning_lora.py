"""Laboratorio 4: demo di SFT LoRA con Qwen3-0.6B usando TRL + PEFT.
PRIMA: fermare vLLM se usa la stessa GPU. NON equivale a fine-tuning di produzione.
Vedi requirements-train.txt. Questo script non è stato allenato sul tuo hardware.
"""
import json
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen3-0.6B"
ROOT = Path(__file__).resolve().parents[1]
if not torch.cuda.is_available():
    raise SystemExit("GPU CUDA assente: non avviare training. Controlla nvidia-smi e PyTorch.")

# Esempi sintetici: solo dimostrazione meccanica del fine-tuning.
train_pairs = [
    ("Che cosa indica GPU?", "GPU indica Graphics Processing Unit."),
    ("Che cos'è un embedding?", "Un embedding è una rappresentazione numerica vettoriale di un contenuto."),
    ("Definisci RAG.", "RAG significa Retrieval-Augmented Generation: recupera fonti esterne prima della generazione."),
    ("Che cos'è vLLM?", "vLLM è un inference engine per servire modelli linguistici."),
    ("Cos'è una KV cache?", "La KV cache conserva rappresentazioni Key e Value dei token già elaborati."),
    ("Cos'è un tool per un agente?", "Un tool è una funzione esterna che l'applicazione può eseguire su richiesta del modello."),
    ("Definisci LoRA.", "LoRA addestra matrici di adattamento a basso rango mantenendo congelato il modello di base."),
    ("Definisci QLoRA.", "QLoRA combina pesi di base quantizzati e adattatori addestrabili."),
    ("Che cos'è HITL?", "Human-in-the-Loop è una revisione umana inserita prima o durante un'azione automatizzata."),
    ("Che cos'è TTFT?", "TTFT è il tempo dall'invio della richiesta al primo token prodotto."),
    ("Che cos'è un vector DB?", "Un vector database conserva vettori e metadati per la ricerca per similarità."),
    ("Che cosa misura HitRate@k?", "HitRate@k misura la quota di domande per cui un documento atteso compare tra i primi k recuperati."),
]
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def make_text(pair):
    messages = [
        {"role":"system", "content":"Rispondi in italiano in una frase tecnica concisa."},
        {"role":"user", "content":pair[0]},
        {"role":"assistant", "content":pair[1]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                         enable_thinking=False)

train = Dataset.from_list([{"text":make_text(item)} for item in train_pairs[:10]])
eval_data = Dataset.from_list([{"text":make_text(item)} for item in train_pairs[10:]])
peft_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)
use_bf16 = torch.cuda.is_bf16_supported()
args = SFTConfig(
    output_dir=str(ROOT / "dati" / "adapter_lora_demo"),
    dataset_text_field="text", max_length=384, packing=False,
    learning_rate=2e-4, max_steps=10, per_device_train_batch_size=1,
    gradient_accumulation_steps=2, logging_steps=1, save_strategy="no",
    report_to="none", bf16=use_bf16, fp16=not use_bf16,
)
trainer = SFTTrainer(model=MODEL, args=args, train_dataset=train,
                     eval_dataset=eval_data, peft_config=peft_cfg,
                     processing_class=tokenizer)
trainer.train()
metrics = trainer.evaluate()
trainer.model.save_pretrained(str(ROOT / "dati" / "adapter_lora_demo"))
print("Adapter salvato. Metriche di valutazione: ", json.dumps(metrics, indent=2))
print("ATTENZIONE: dataset troppo piccolo per giudizi di qualità.")
