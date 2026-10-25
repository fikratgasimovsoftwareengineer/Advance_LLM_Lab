# Videotutorial verificati e riferimenti tecnici (consultati ottobre 2026)

I video sono ospitati sulle rispettive piattaforme esterne. Alcuni richiedono la creazione di un account o applicano condizioni di accesso proprie; non sono inclusi come file nel kit.

| Modulo | Videocorso | Durata indicativa | URL |
|---|---|---:|---|
| 1 | How Transformer LLMs Work — Jay Alammar / Maarten Grootendorst | 1h34 | https://www.deeplearning.ai/courses/how-transformer-llms-work |
| 2 | Fast & Efficient LLM Inference with vLLM — Cedric Clyburn | 1h38 | https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm |
| 2/7 | Efficiently Serving LLMs — Travis Addair | 2h30 | https://www.deeplearning.ai/courses/efficiently-serving-llms |
| 3 | Building and Evaluating Advanced RAG — Jerry Liu / Anupam Datta | 1h55 | https://www.deeplearning.ai/courses/building-evaluating-advanced-rag |
| 4 | Finetuning Large Language Models — Sharon Zhou | 1h25 | https://www.deeplearning.ai/courses/finetuning-large-language-models |
| 5/6 | Introduction to LangGraph — LangChain Academy | ~6h | https://academy.langchain.com/courses/intro-to-langgraph |
| 5/6 | AI Agents in LangGraph — Harrison Chase / Rotem Weiss | 1h32 | https://www.deeplearning.ai/courses/ai-agents-in-langgraph |
| 6 | Safe and Reliable AI via Guardrails — Shreya Rajpal | 1h32 | https://www.deeplearning.ai/courses/safe-and-reliable-ai-via-guardrails |
| 7 | Building toward Computer Use with Anthropic — lezione Prompt Caching | 12m per lezione | https://www.deeplearning.ai/courses/building-toward-computer-use-with-anthropic |

### Fonti tecniche
- vLLM 0.31.0, API, caratteristiche e benchmark: https://docs.vllm.ai/en/v0.31.0/
- vLLM automatic prefix caching: https://docs.vllm.ai/en/v0.31.0/features/automatic_prefix_caching/
- Hugging Face TRL + PEFT: https://huggingface.co/docs/trl/peft_integration
- Hugging Face AI Agents: https://huggingface.co/learn/agents-course/en/unit0/introduction
- Agentic RAG: https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/agentic-rag
- LangGraph course: https://academy.langchain.com/courses/intro-to-langgraph
- Amazon Bedrock Prompt Caching: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
- OWASP LLM Prompt Injection: https://genai.owasp.org/llmrisk/llm01-prompt-injection/

### Nota di aggiornamento
I comandi in tutorial pubblicati in momenti diversi possono riferirsi a versioni differenti. Per il tuo ambiente usare come riferimento operativo `vllm --version` e `vllm serve --help`. Esempi adatti a Qwen3-0.6B non sono automaticamente validi per ogni altro modello.
