# 
# # Core dependencies (usually pre-installed in Colab, but ensure updated)
# !pip install --upgrade transformers pandas tqdm torch

# # (Optional) If vllm fails due to CUDA or compatibility, you may need a GPU-optimized runtime
# !pip install PyMuPDF pdfminer.six
# !pip uninstall -y torch torchaudio torchvision
# !pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu121

# # Install a potentially more compatible version of vllm for older GPUs
# !pip install vllm==0.2.7

import re
from collections import Counter
import vllm
from vllm import LLM
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import set_seed
from collections import Counter
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import set_seed
import json, os
import logging


# for text_Extract
import os
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd


# for token and resource counting
import tiktoken
import time
import psutil



model_id = "meta-llama/Llama-3.1-8B"

llm = LLM(
    model=model_id,
    trust_remote_code=True,
    max_model_len=32768,   # try 16k; should be safer than putting full 32‑128k
    enable_prefix_caching=True,
    tensor_parallel_size=torch.cuda.device_count(),  # likely =1
    dtype="float16",   # vLLM may still need a higher precision dtype for non‑quantized parts
)

tokenizer = llm.get_tokenizer()


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    bold_yellow = "\x1b[33;1m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_green = "\x1b[32;20;1m"
    bold_red = "\x1b[31;1m"
    bold_white = "\x1b[37;1m"
    orange = "\x1b[38;5;214m"
    bold_orange = "\x1b[38;5;214;1m"
    reset = "\x1b[0m"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: format,
        logging.WARNING: bold_yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
        31: reset + format + reset,
        32: green + format + reset,
        33: bold_green + format + reset,
        34: bold_white + format + reset,
        35: orange + format + reset,
        36: bold_orange + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(__name__)
logger.propagate = False
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


# def llm_engine(messages, stop_sequences=None, start_sequence=None) -> str:
#     sampling_params = vllm.SamplingParams(
#         temperature=0.7,
#         top_p=0.9,
#         # use_beam_search=True,
#         # num_beams=3,
#         best_of=1,
#         max_tokens=32768,
#         stop=stop_sequences,
#         include_stop_str_in_output=True,
#     )
#     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     if start_sequence:
#         prompt += start_sequence
#     output = llm.generate([prompt], sampling_params, use_tqdm=False)
#     response = output[0].outputs[0].text

#     if start_sequence:
#         response = start_sequence + response
#     return response



# for llama3.1-8B, ensure tokenizer has a chat_template
from transformers import ChatTemplate
def llm_engine(messages, stop_sequences=None, start_sequence=None) -> str:
    # Ensure tokenizer has a chat_template
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        
        tokenizer.chat_template = ChatTemplate(
            system="<system>{system}</system>\n",
            user="<user>{user}</user>\n",
            assistant="<assistant>{assistant}</assistant>\n"
        )

    # Convert messages into a single prompt string
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Append start_sequence if provided
    if start_sequence:
        prompt += start_sequence

    # Set sampling parameters for vLLM
    sampling_params = vllm.SamplingParams(
        temperature=0.7,
        top_p=0.9,
        best_of=1,
        max_tokens=32768,
        stop=stop_sequences,
        include_stop_str_in_output=True,
    )

    # Generate output
    output = llm.generate([prompt], sampling_params, use_tqdm=False)
    response = output[0].outputs[0].text

    # Prepend start_sequence to response if needed
    if start_sequence:
        response = start_sequence + response

    return response


def extract_answer(response):
    # Regex pattern to match content inside \boxed{...}
    pattern = r'\\boxed{(-?\d+)}'

    # Search for the match
    match = re.search(pattern, response)

    if match:
        answer = int(match.group(1))  # Get the content inside the curly braces
    else:
        answer = -1
    return answer


# def cot_sc(question: str, num_paths=16):
#     sampling_params = vllm.SamplingParams(
#         n=num_paths,
#         temperature=0.7,
#         top_p=0.8,
#         repetition_penalty=1.05,
#         max_tokens=8192
#     )

#     prompt = question
#     messages = [
#         {"role": "system", "content": "Please reason step by step in English, and put your final answer within \\boxed{}."},
#         {"role": "user", "content": prompt}
#     ]

#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     outputs = llm.generate([text], sampling_params, use_tqdm=False)
#     outputs = [output.text for output in outputs[0].outputs]
#     answers = [extract_answer(output) for output in outputs]
#     answers = [answer for answer in answers if answer >= 0]

#     if answers:
#         answer, _ = Counter(answers).most_common(1)[0]
#     else:
#         answer = 0

#     return answer


SYSTEM_PROMPT = """You are an expert system designed to answer questions based on TriviaQA-style (JSONL, similar to NQ) datasets.

For each question-answer-context instance, follow this workflow and output using the specified tags:

**Thought Process**  
   - Wrap in `<thought>` tags.  
   - This is your internal reasoning before producing the final structured fields.  
   - Mention how you analyzed the context, located the answer, and verified it.  
   Example:  
   <thought>The context mentions...</thought>

1. **question_id**  
   - Wrap in `<id>` tags.  
   Example: <id>123</id>

2. **question**  
   - Wrap in `<question>` tags.  
   Example: <question>Who discovered penicillin?</question>

3. **answer**  
   - Wrap in `<answer>` tags.  
   - This should be the concise, factual answer from the context.  
   Example: <answer>Alexander Fleming</answer>

4. **context**  
   - Wrap in `<context>` tags.  
   - Provide the exact context passage from the dataset.  
   Example: <context>Penicillin was discovered in 1928 by Alexander Fleming at St. Mary’s Hospital, London.</context>

5. **response**  
   - Wrap in `<response>` tags.  
   - This is the reasoning chain + explanation of how the answer was derived from the context.  
   Example: <response>The context explicitly mentions Alexander Fleming as the discoverer of penicillin in 1928.</response>

6. **response_length**  
   - Wrap in `<response_length>` tags.  
   - Give word count (or token count) of the response.  
   Example: <response_length>14</response_length>

7. **Faithfulness**  
   - Wrap in `<Faithfulness>` tags.  
   - Judge if the answer strictly follows the context: between (0 to 1).  
   Example: <Faithfulness>1</Faithfulness>

8. **Completeness**  
   - Wrap in `<Completeness>` tags.  
   - Does the answer fully address the question? between (0 to 1).  
   Example: <Completeness>1</Completeness>

9. **Answer Relevance**  
   - Wrap in `<Answer_Relevance>` tags.  
   - Is the answer relevant to the asked question? between (0 to 1).  
   Example: <Answer_Relevance>1</Answer_Relevance>

10. **Context Relevance**  
   - Wrap in `<Context_Relevance>` tags.  
   - Does the given context directly support the answer? between (0 to 1).  
   Example: <Context_Relevance>1</Context_Relevance>

11. **Context Recall**  
   - Wrap in `<Context_Recall>` tags.  
   - Assess how well the system recalled the context in its reasoning: between (0 to 1).  
   Example: <Context_Recall>1</Context_Recall>

"""




from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer




import re
import logging

logger = logging.getLogger(__name__)

class ReActRAG:
    def __init__(self, llm_engine, max_iterations=1):
        self.llm_engine = llm_engine
        self.max_iterations = max_iterations

        # Compile regex patterns for new tags
        self.patterns = {
            "thought": re.compile(r"<thought>(.*?)</thought>", re.DOTALL),
            "id": re.compile(r"<id>(.*?)</id>", re.DOTALL),
            "question": re.compile(r"<question>(.*?)</question>", re.DOTALL),
            "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL),
            "context": re.compile(r"<context>(.*?)</context>", re.DOTALL),
            "response": re.compile(r"<response>(.*?)</response>", re.DOTALL),
            "response_length": re.compile(r"<response_length>(.*?)</response_length>", re.DOTALL),
            "Faithfulness": re.compile(r"<Faithfulness>(.*?)</Faithfulness>", re.DOTALL),
            "Completeness": re.compile(r"<Completeness>(.*?)</Completeness>", re.DOTALL),
            "Answer_Relevance": re.compile(r"<Answer_Relevance>(.*?)</Answer_Relevance>", re.DOTALL),
            "Context_Relevance": re.compile(r"<Context_Relevance>(.*?)</Context_Relevance>", re.DOTALL),
            "Context_Recall": re.compile(r"<Context_Recall>(.*?)</Context_Recall>", re.DOTALL),
            "Exact_Match": re.compile(r"<Exact_Match>(.*?)</Exact_Match>", re.DOTALL),

        }

    def run(self, task: str):
        system_message = {"role": "system", "content": SYSTEM_PROMPT}
        task_message = {"role": "user", "content": task}
        messages = [system_message, task_message]

        logger.log(33, "======== New Q&A Task ========")
        logger.log(34, task)

        extracted_data = {}

        for _ in range(self.max_iterations):
            response = self.llm_engine(
                messages,
                stop_sequences=["</Context_Recall>"],  # stop after last tag
                start_sequence="<thought>\n"
            )

            # Extract fields
            for key, pattern in self.patterns.items():
                matches = pattern.findall(response)
                if matches:
                    extracted_data[key] = matches[0].strip()

            # Logging all extracted fields
            if "thought" in extracted_data:
                logger.log(35, "=== Agent Thought ===")
                logger.log(31, extracted_data["thought"])

            if "id" in extracted_data:
                logger.log(32, "=== ID ===")
                logger.log(31, extracted_data["id"])

            if "question" in extracted_data:
                logger.log(32, "=== Question ===")
                logger.log(31, extracted_data["question"])

            if "answer" in extracted_data:
                logger.log(33, "=== Answer ===")
                logger.log(32, extracted_data["answer"])

            if "context" in extracted_data:
                logger.log(32, "=== Context ===")
                logger.log(31, extracted_data["context"])

            if "response" in extracted_data:
                logger.log(36, "=== Agent Response ===")
                logger.log(31, extracted_data["response"])

            if "response_length" in extracted_data:
                logger.log(32, "=== Response Length ===")
                logger.log(31, extracted_data["response_length"])

            if "Faithfulness" in extracted_data:
                logger.log(32, "=== Faithfulness ===")
                logger.log(31, extracted_data["Faithfulness"])

            if "Completeness" in extracted_data:
                logger.log(32, "=== Completeness ===")
                logger.log(31, extracted_data["Completeness"])

            if "Answer_Relevance" in extracted_data:
                logger.log(32, "=== Answer Relevance ===")
                logger.log(31, extracted_data["Answer_Relevance"])

            if "Context_Relevance" in extracted_data:
                logger.log(32, "=== Context Relevance ===")
                logger.log(31, extracted_data["Context_Relevance"])
            if "Exact_Match" in extracted_data:
                logger.log(32, "=== Exact Match ===")
                logger.log(31, extracted_data["Exact_Match"])

            if "Context_Recall" in extracted_data:
                logger.log(32, "=== Context Recall ===")
                logger.log(31, extracted_data["Context_Recall"])
                break

            if not extracted_data:
                logger.error("Agent did not return complete output. Retrying...")

            messages.append({"role": "assistant", "content": response})

        else:
            logger.error("Reached max iterations without a final answer.")
            return None

        # Return only the 11 CSV-compatible fields
        return {
            "id": extracted_data.get("id"),
            "question": extracted_data.get("question"),
            "answer": extracted_data.get("answer"),
            "context": extracted_data.get("context"),
            "response": extracted_data.get("response"),
            "response_length": extracted_data.get("response_length"),
            "Faithfulness": extracted_data.get("Faithfulness"),
            "Completeness": extracted_data.get("Completeness"),
            "Answer_Relevance": extracted_data.get("Answer_Relevance"),
            "Context_Relevance": extracted_data.get("Context_Relevance"),
            "Context_Recall": extracted_data.get("Context_Recall"),
            "Exact_Match": extracted_data.get("Exact_Match"),
        }


# Example instantiation
agent = ReActRAG(
    llm_engine=llm_engine,
    max_iterations=1,
)


# import json

# def convert_triviaqa_sample_to_text(json_path, include_context=True, max_context_len=1000):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         sample = json.load(f)

#     question = sample.get("question", "").strip()
#     answer = sample.get("answer", {}).get("value", "N/A")

#     # Extract context
#     context = ""
#     if include_context:
#         descriptions = sample.get("search_results", {}).get("description", [])
#         search_contexts = sample.get("search_results", {}).get("search_context", [])
        
#         context_blocks = descriptions + search_contexts
#         context = "\n".join(context_blocks).strip()

#         # Optional: truncate context if too long
#         if max_context_len and len(context) > max_context_len:
#             context = context[:max_context_len] + "..."

#     # Format output
#     if context:
#         extracted_text = f"Context:\n{context}\n\nQ: {question}\nA: {answer}"
#     else:
#         extracted_text = f"Q: {question}\nA: {answer}"

#     return extracted_text


# # # #


import json
import os
import pandas as pd
from tqdm import tqdm
import logging

# Optional: define your logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def convert_triviaqa_sample_to_text(sample, include_context=True, max_context_len=5000):
    """Convert a single TriviaQA JSON sample into a readable text format including all keys."""

    def flatten_dict(d, parent_key=''):
        """Recursively flattens a nested dictionary."""
        items = []
        for k, v in d.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, full_key))
            elif isinstance(v, list):
                if v:
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            items.extend(flatten_dict(item, f"{full_key}[{i}]"))
                        else:
                            items.append((f"{full_key}[{i}]", item))
                else:
                    items.append((full_key, "[]"))
            else:
                items.append((full_key, v if v != "" else '""'))
        return items

    # Flatten the full sample to get all keys and values
    flat_items = flatten_dict(sample)

    # Extract question and answer separately for easier access
    question = sample.get("question", "").strip()
    answer = sample.get("answer", {}).get("value", "N/A")

    # Optional: extract + truncate context (same logic as before)
    context = ""
    if include_context:
        descriptions = sample.get("search_results", {}).get("description", [])
        search_contexts = sample.get("search_results", {}).get("search_context", [])

        context_blocks = descriptions + search_contexts
        context = "\n".join(context_blocks).strip()

        if max_context_len and len(context) > max_context_len:
            context = context[:max_context_len].rsplit("\n", 1)[0] + "\n[...]"

    # Build full extracted text with all key-value pairs
    all_fields_text = "\n".join([f"{k}: {v}" for k, v in flat_items])

    if context:
        final_text = f"{all_fields_text}\n\n---\nContext:\n{context}\n\nQ: {question}\nA: {answer}"
    else:
        final_text = f"{all_fields_text}\n\n---\nQ: {question}\nA: {answer}"

    return final_text, question


def safe_run(agent, task, retries=25):
    for attempt in range(retries):
        response = agent.run(task)
        if isinstance(response, dict) and response.get("answer", "").strip():
            return response
        logger.warning(f"Empty response on attempt {attempt+1}, retrying...")
    return ""  # fallback after retries

# === MAIN SCRIPT ===
import re
import time
import psutil

def extract_field(response_text, field_name):
    """
    Naive field extractor based on line starting with `field_name:`
    """
    pattern = rf"{field_name}\s*[:\-]\s*(.*)"
    match = re.search(pattern, response_text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


# Load JSONL properly (not using pandas.read_csv for JSONL)
jsonl_path = "dataset/squad_v2_subset_1000.jsonl"
with open(jsonl_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

results = []

for i, sample in tqdm(enumerate(data), total=len(data)):
    try:
        # Convert sample to prompt text
        extracted_text, question = convert_triviaqa_sample_to_text(sample)
        encoding = tiktoken.get_encoding("gpt2")
        
        # Measure latency
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # in MB
        

        # Extract gold answer from sample for exact match computation
        answer = sample.get("answer", {}).get("value", "N/A")

        # Create prompt
        user_prompt = f"""
        Answer the following question using only the provided context. Your answer must be based solely on the information in the context. Do not use any external knowledge or make assumptions.

        Your answer should contain
        1. id
        2. question
        3. answer
        4. context
        5. response
        6. response_length
        7. Faithfulness
        8. Completeness
        9. Answer Relevance
        10. Context Relevance
        11. Context Recall


        {extracted_text}

        """.strip()

        # generated_tokens = encoding.encode(user_prompt)
        generated_tokens = len(encoding.encode(user_prompt))
        

        # Run agent with retry logic
        response = safe_run(agent, user_prompt, retries=25)
        response_str = str(response)

        response_tokens = len(encoding.encode(response_str))

            # Measure latency
        latency = time.time() - start_time
        end_mem = process.memory_info().rss
        resource_used = end_mem - start_mem
        
        # Compute Exact Match
        import re, unicodedata, string

        def _normalize(s):
            if s is None:
                return ""
            s = unicodedata.normalize("NFKC", str(s))
            s = s.lower()
            s = re.sub(r'\s+', ' ', s).strip()
            # remove punctuation
            s = s.translate(str.maketrans('', '', string.punctuation))
            # remove common English articles (optional)
            s = re.sub(r'\b(a|an|the)\b', '', s).strip()
            return s

        def compute_exact_match(predicted, gold):
            """
            gold may be a single string or an iterable of acceptable gold strings.
            Returns 1 for exact match after normalization, else 0.
            """
            pred_norm = _normalize(predicted)
            if isinstance(gold, (list, tuple, set)):
                return int(any(pred_norm == _normalize(g) for g in gold))
            return int(pred_norm == _normalize(gold))


        # Example: extracting model's answer (customize based on actual output format)
        model_answer = extract_field(response_str, "answer")  # <-- You must define this helper
        exact_match = compute_exact_match(model_answer, answer)

        


        if response is not None:
            results.append({
                "id": response.get("id", i),  # fallback to loop index if missing
                "question": response.get("question", ""),
                "answer": response.get("answer", ""),
                "context": response.get("context", ""),
                "response": response.get("response", ""),
                "response_length": response.get("response_length", ""),
                "Faithfulness": response.get("Faithfulness", ""),
                "Completeness": response.get("Completeness", ""),
                "Answer_Relevance": response.get("Answer_Relevance", ""),
                "Context_Relevance": response.get("Context_Relevance", ""),
                "Context_Recall": response.get("Context_Recall", ""),
                "exact_match": exact_match,
                "generated_tokens": generated_tokens,
                "response_tokens": response_tokens,
                "latency": latency,
                "resource_used": resource_used
            })
        else:
            logger.error(f"Agent returned None for sample {i}")
            results.append({
                "id": i,
                "question": "",
                "answer": "",
                "context": "",
                "response": "",
                "response_length": "",
                "Faithfulness": "",
                "Completeness": "",
                "Answer_Relevance": "",
                "Context_Relevance": "",
                "Context_Recall": "",
                "exact_match": exact_match,
                "generated_tokens": generated_tokens,
                "response_tokens": response_tokens,
                "latency": latency,
                "resource_used": resource_used
            })

    except Exception as e:
        logger.error(f"Failed to process sample {i}: {e}")
        results.append({
            "id": i,
            "question": "",
            "answer": "",
            "context": "",
            "response": "",
            "response_length": "",
            "Faithfulness": "",
            "Completeness": "",
            "Answer_Relevance": "",
            "Context_Relevance": "",
            "Context_Recall": "",
            "exact_match": exact_match,
            "generated_tokens": generated_tokens,
            "response_tokens": response_tokens,
            "latency": latency,
            "resource_used": resource_used
        })

#csv output
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("ReActRAG_v4_Lllama3.1-8B_squad.csv", index=False, encoding="utf-8")

print(f"Wrote ReActRAG_v4_Lllama3.1-8B_squad.csv with {len(results)} rows.")

# json output

with open("ReActRAG_v4_Lllama3.1-8B_squad.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Wrote ReActRAG_v4_Lllama3.1-8B_squad.json with {len(results)} rows.")

