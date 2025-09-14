from datasets import load_dataset
from itertools import islice

# Load SQuAD v2 in streaming mode
streamed_dataset = load_dataset("squad_v2", split="train", streaming=True)

# Take the first 500 examples from the stream
subset = list(islice(streamed_dataset, 500))

# Optional: Save to JSON Lines file
import json

with open("squad_v2_subset_500.jsonl", "w") as f:
    for example in subset:
        f.write(json.dumps(example) + "\n")

# Show first example
print(subset[0])
