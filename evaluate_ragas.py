# Ragas evaluation script for TriviaQA predictions
import pandas as pd
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas import evaluate

# Load predictions
df = pd.read_csv("Result_SquaD/ReActRAG_v5_Qwen3-8B_SQuAD.csv")

# Prepare Ragas input
ragas_data = [
    {
        "question": row["question"],
        "contexts": [row["context"]],
        "ground_truth": row["answer"],
        # "prediction": row["prediction"]
    }
    for _, row in df.iterrows()
]

# Evaluate
results = evaluate(
    ragas_data,
    metrics=[answer_relevancy(), faithfulness(), context_precision(), context_recall()]
)

print("Ragas Evaluation Results:")
for metric, score in results.items():
    print(f"{metric}: {score}")
