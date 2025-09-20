from ragas.metrics import faithfulness, answer_relevance, context_recall
from ragas import evaluate
from datasets import Dataset
from langchain_community.llms import VLLM
import pandas as pd

# Load dataset (like I showed earlier)
df = pd.read_csv("Result_TriviaQA/ReActRAG_v5_Qwen3-8B_SQuAD.csv")
df["contexts"] = df["context"].apply(lambda x: [x] if isinstance(x, str) else x)
ragas_df = Dataset.from_pandas(df[["question", "answer", "contexts", "response"]])

# Setup vLLM as judge
llm = VLLM(model="Qwen/Qwen3-8B", tensor_parallel_size=1)

# Evaluate
results = evaluate(
    ragas_df,
    metrics=[faithfulness, answer_relevance, context_recall],
    llm=llm
)
print(results)
