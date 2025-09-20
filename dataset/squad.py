from datasets import load_dataset
from itertools import islice

# Load SQuAD v2 in streaming mode
streamed_dataset = load_dataset("squad_v2", split="train", streaming=True)

# Take the first 500 examples from the stream
subset = list(islice(streamed_dataset, 20000))

# Optional: Save to JSON Lines file
import json

with open("squad_v2_subset_20000.jsonl", "w") as f:
    for example in subset:
        f.write(json.dumps(example) + "\n")

# Show first example
print(subset[0])


# csv


# from datasets import load_dataset
# from itertools import islice
# import csv

# # Load SQuAD v2 training set in streaming mode
# streamed_dataset = load_dataset("squad_v2", split="train", streaming=True)

# # Take the first 500 examples
# subset = list(islice(streamed_dataset, 500))

# # Define which fields to keep in the CSV
# fields = ["id", "title", "context", "question", "answers"]

# # Write to CSV
# with open("squad_v2_subset_500.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=fields)
#     writer.writeheader()
#     for example in subset:
#         # Convert 'answers' (which is a dict with lists) to string
#         example["answers"] = str(example["answers"])
#         writer.writerow({field: example[field] for field in fields})
