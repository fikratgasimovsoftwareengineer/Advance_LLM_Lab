from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)

from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel


judge = OllamaModel(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0
)

test_case = LLMTestCase(
    input = "Entro quanti giorni posso procedere?",
    actual_output="il processo e` consentito entro 14 giorni",
    retrieval_context=[
        "il cliente puo` esercitare il diritto",
        "il diritto entro 14 giorni"
    ],
)

metrices = [
    FaithfulnessMetric(
        threshold=0.8, model=judge
    ),
    AnswerRelevancyMetric(
        threshold=0.8, model=judge
        
    ),
    ContextualRelevancyMetric(
        threshold=0.8, model=judge
    )
]

for metric in metrices:
    metric.measure(test_case)
    
    print(type(metric).__name__)
    print("score: ", metric.score)
    print("passed: ", metric.is_successful())
    print("reason", metric.reason)