import argparse
import gc
import json
import logging
import os
import random
import re
import shutil
from math import ceil
from typing import Literal

import PyPDF2
import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("raft_script")

# Global flag to force CPU-only mode
FORCE_CPU_ONLY = False

# Document type literals
DocType = Literal["api", "pdf", "json", "txt", "md"]

# Every N chunks, save a checkpoint
N = 15


def get_device():
    """
    Determines the appropriate device (CUDA/CPU) to use based on availability and settings.
    """
    if FORCE_CPU_ONLY or not torch.cuda.is_available():
        logger.info("Using CPU for model inference")
        return torch.device("cpu")
    logger.info("Using CUDA for model inference")
    return torch.device("cuda")


def load_model(model_name: str):
    """
    Loads a model with caching for better memory management.

    Args:
        model_name: The Hugging Face model name

    Returns:
        The loaded model and tokenizer
    """
    device = get_device()

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get model
    logger.info(f"Loading model {model_name} to {device}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    input_text: str,
    max_length: int = 512,
    num_return_sequences: int = 1,
    temperature: float = 0.7,
) -> list[str]:
    """
    Generates text using a transformer model.

    Args:
        model: The model
        tokenizer: The tokenizer
        input_text: The input text prompt
        max_length: Maximum length of the generated text
        num_return_sequences: Number of sequences to generate
        temperature: Temperature for sampling

    Returns:
        A list of generated text sequences
    """
    try:
        device = next(model.parameters()).device

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="longest").to(device)

        do_sample = temperature > 0

        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
        )

        result = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Clean up to free memory
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return result
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return ["Failed to generate text due to an error."] * num_return_sequences


def get_args() -> argparse.Namespace:
    """
    Parses and returns the command line arguments specified by the user.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", type=str, default="", help="The path at which the document is located")
    parser.add_argument("--output", type=str, default="./", help="The path at which to save the dataset")
    parser.add_argument(
        "--output-format", type=str, default="hf", help="Format to convert the dataset to. Defaults to hf."
    )
    parser.add_argument(
        "--output-type", type=str, default="jsonl", help="Type to export the dataset to. Defaults to jsonl."
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=3,
        help="The number of distractor documents to include per data point / triplet",
    )
    parser.add_argument(
        "--p", type=float, default=1.0, help="The percentage that the oracle document is included in the context"
    )
    parser.add_argument(
        "--questions", type=int, default=5, help="The number of data points / triplets to generate per chunk"
    )
    parser.add_argument("--chunk_size", type=int, default=1024, help="The size of each chunk in number of tokens")
    parser.add_argument(
        "--doctype",
        type=str,
        default="pdf",
        help="The type of the document",
        choices=["pdf", "txt", "json", "api", "md"],
    )
    parser.add_argument("--fast", action="store_true", help="Run the script in fast mode (no recovery implemented)")
    parser.add_argument(
        "--qg-model", type=str, default="google/flan-t5-large", help="Model to use for question generation"
    )
    parser.add_argument(
        "--cot-model", type=str, default="google/flan-t5-xl", help="Model to use for chain-of-thought reasoning"
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only mode for model inference")

    args = parser.parse_args()

    # Set global CPU-only flag
    global FORCE_CPU_ONLY
    FORCE_CPU_ONLY = args.cpu_only
    if FORCE_CPU_ONLY:
        logger.info("CPU-only mode enabled. Will not use GPU even if available.")

    return args


def get_chunks(file_path: str, doctype: DocType = "pdf", chunk_size: int = 1024) -> list[str]:
    """
    Takes in a `file_path` and `doctype`, retrieves the document, breaks it down into chunks of size
    `chunk_size`, and returns the chunks as a list of strings.
    """
    chunks = []

    logger.info(f"Retrieving chunks from {file_path} of type {doctype}")

    if doctype == "api":
        # Load API documentation and process it
        with open(file_path) as f:
            api_docs_json = json.load(f)
        chunks = [str(api_doc_json) for api_doc_json in api_docs_json]

    else:
        if doctype == "json":
            # Load JSON document
            with open(file_path) as f:
                data = json.load(f)
            text = data["text"]
        elif doctype == "pdf":
            # Load PDF and extract text
            text = ""
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    text += page.extract_text()
        elif doctype == "txt" or doctype == "md":
            # Load plain text document
            with open(file_path, encoding="utf-8") as file:
                text = file.read()
        else:
            raise TypeError("Document is not one of the accepted types: api, pdf, json, txt, md")

        # For RPG content, try to create meaningful chunks based on headers or sections
        if doctype == "md":
            # Try to split on markdown headers
            chunks = re.split(r"\n##? ", text)
            # If chunks are too large, further split them
            new_chunks = []
            for chunk in chunks:
                if len(chunk) > chunk_size:
                    for i in range(0, len(chunk), chunk_size):
                        new_chunks.append(chunk[i : i + chunk_size])
                else:
                    new_chunks.append(chunk)
            chunks = new_chunks
        else:
            # Split the text into chunks of roughly equal size
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i : i + chunk_size])

    return chunks


def clean_chunk(chunk: str) -> str:
    """
    Cleans a chunk of text to make it more suitable for question generation.
    Removes excessive whitespace, special characters, etc.
    """
    # Replace multiple newlines with a single one
    chunk = re.sub(r"\n+", "\n", chunk)
    # Replace multiple spaces with a single one
    chunk = re.sub(r" +", " ", chunk)
    # Remove special characters that might confuse the model
    chunk = re.sub(r"[^\w\s\.,;:!?\-\'\"()\[\]{}]", "", chunk)
    return chunk.strip()


def generate_questions_hf(chunk: str, x: int = 5, model_name: str = "google/flan-t5-large") -> list[str]:
    """
    Uses PyTorch model to generate `x` questions based on the given text chunk.

    Args:
        chunk: The text chunk to generate questions from
        x: Number of questions to generate
        model_name: The model to use for generation

    Returns:
        A list of generated questions
    """
    try:
        # Load model and tokenizer
        model, tokenizer = load_model(model_name)
        logger.info(f"Generating {x} questions with model {model_name}")

        # Clean the chunk
        clean_text = clean_chunk(chunk)

        # Prepare prompt for better question generation
        input_text = f"""Generate {x} insightful questions about the following text. The
text is taken from documents describing a fictional world or setting used for a tabletop rpg game.
The text may contain references to characters, locations, events, or rules, as well as other documents
or sources. Do not confuse these other documents with descriptions of books or other writings that appear
within the fitional world. The questions should cover different aspects of the text and require understanding to answer.
Include some questions that need reasoning across multiple parts of the text.

Text: {clean_text}

Questions:"""

        # Generate questions
        outputs = generate_text(model, tokenizer, input_text, max_length=256, num_return_sequences=x, temperature=0.8)

        # Clean up resources to free memory
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Clean up the generated questions
        clean_questions = []
        for q in outputs:
            # Extract just the question part if numbered
            if re.match(r"^\d+[\.\)]\s", q):
                q = re.sub(r"^\d+[\.\)]\s", "", q)
            # Make sure it ends with a question mark
            if not q.endswith("?"):
                q += "?"
            clean_questions.append(q)

        return clean_questions[:x]  # Ensure we return exactly x questions
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        # Return placeholder questions if generation fails
        return [f"Question {i + 1} about the text?" for i in range(x)]


def generate_cot_answer(question: str, oracle_context: str, model_name: str = "google/flan-t5-xl") -> dict:
    """
    Generates a chain-of-thought answer with reasoning and citations from the context using PyTorch.

    Args:
        question: The question to answer
        oracle_context: The context text containing the answer
        model_name: The model to use for generation

    Returns:
        A dictionary with reasoning and final answer
    """
    try:
        # Load model and tokenizer
        model, tokenizer = load_model(model_name)

        # Create a prompt that encourages chain-of-thought reasoning with citations
        prompt = f"""Answer the following question based on the given context.
The context is taken from documents describing a fictional world or setting used for a tabletop rpg game.
It contain references to characters, locations, events, or rules, as well as other documents
or sources. Do not confuse these other documents with descriptions of books or other writings that appear
within the fitional world.
First, identify relevant information from the context with direct quotes.
Then, explain your reasoning step by step.
Finally, provide a concise final answer.

Context: {oracle_context}

Question: {question}

Answer:"""

        # Generate answer
        outputs = generate_text(model, tokenizer, prompt, max_length=512, temperature=0.7)

        # Clean up resources to free memory
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        raw_answer = outputs[0]

        # Process the answer to extract reasoning and final answer
        try:
            # Try to identify reasoning and final answer sections
            reasoning_pattern = r"(?:.*?(?:According to the context|From the provided information|Based on the context|Looking at the context))(.*?)(?:Therefore,|In conclusion,|To summarize,|Thus,|So,|Final answer:|The answer is:)(.*)"  # noqa: E501
            match = re.search(reasoning_pattern, raw_answer, re.DOTALL | re.IGNORECASE)

            if match:
                reasoning = match.group(1).strip()
                final_answer = match.group(2).strip()
            else:
                # If pattern not found, use a simpler split
                lines = raw_answer.split("\n")
                if len(lines) >= 3:
                    # Assume last line or two is the final answer
                    final_answer = "\n".join(lines[-2:])
                    reasoning = "\n".join(lines[:-2])
                else:
                    reasoning = raw_answer
                    final_answer = raw_answer

            # Insert quote markers around any direct quotes
            reasoning_with_quotes = re.sub(r'"([^"]+)"', r"##begin_quote##\1##end_quote##", reasoning)

            return {"reasoning": reasoning_with_quotes, "final_answer": final_answer}

        except Exception as e:
            logger.warning(f"Error parsing CoT answer: {e}")
            return {"reasoning": "Based on the provided context, " + raw_answer, "final_answer": raw_answer}
    except Exception as e:
        logger.error(f"Error generating CoT answer: {e}")
        # Return a simple fallback answer
        return {
            "reasoning": f"Error generating reasoning: {str(e)}",
            "final_answer": "Unable to generate a proper answer due to an error.",
        }


def add_chunk_to_dataset(
    chunks: list[str],
    chunk: str,
    doctype: DocType = "md",
    x: int = 5,
    num_distract: int = 3,
    p: float = 0.8,
    qg_model: str = "google/flan-t5-large",
    cot_model: str = "google/flan-t5-xl",
) -> None:
    """
    Given a chunk, create {Q, A, D} triplets with chain-of-thought reasoning and add them to the dataset.
    """
    global ds
    i = chunks.index(chunk)

    # Generate questions using the enhanced model
    qs = generate_questions_hf(chunk, x, model_name=qg_model)

    for q in qs:
        datapt = {
            "id": None,
            "type": None,
            "question": None,
            "context": None,
            "oracle_context": None,
            "cot_answer": None,
        }

        datapt["id"] = f"seed_task_{0 if not ds else ds.num_rows}"
        datapt["type"] = "rpg_setting" if doctype == "md" else "general"
        datapt["question"] = q

        # Create distractor documents
        docs = [chunk]
        indices = list(range(0, len(chunks)))
        indices.remove(i)
        for j in random.sample(indices, min(num_distract, len(indices))):
            docs.append(chunks[j])
        # Decide whether to add oracle document
        oracle = random.uniform(0, 1) < p
        if not oracle and indices:
            docs[0] = chunks[random.sample(indices, 1)[0]]
        random.shuffle(docs)

        d = {"title": ["placeholder_title"] * (len(docs)), "sentences": docs}
        datapt["context"] = d
        datapt["oracle_context"] = chunk

        # Generate chain-of-thought answer
        answer_dict = generate_cot_answer(q, chunk, model_name=cot_model)
        datapt["cot_answer"] = answer_dict

        # Construct model instruction
        context = ""
        for idx, doc in enumerate(docs):
            context += f"<DOCUMENT {idx + 1}>\n{str(doc)}\n</DOCUMENT {idx + 1}>\n\n"
        context += q
        datapt["instruction"] = context

        # Add to dataset
        if not ds:
            # Initialize dataset
            datapt["id"] = [datapt["id"]]
            datapt["type"] = [datapt["type"]]
            datapt["question"] = [datapt["question"]]
            datapt["context"] = [datapt["context"]]
            datapt["oracle_context"] = [datapt["oracle_context"]]
            datapt["cot_answer"] = [datapt["cot_answer"]]
            datapt["instruction"] = [datapt["instruction"]]
            ds = Dataset.from_dict(datapt)
        else:
            ds = ds.add_item(datapt)


def save_checkpoint(state, filename):
    """
    Saves the current state of processing to a file for recovery.
    """
    with open(filename, "w") as f:
        f.write(str(state))


def load_checkpoint(filename):
    """
    Loads the processing state from a checkpoint file.
    """
    with open(filename) as f:
        return int(f.read())


def convert_to_jsonl(dataset, output_path):
    """
    Converts the dataset to JSONL format suitable for RAFT training.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            # Convert to RAFT format
            raft_item = {
                "question": item["question"],
                "context": [doc for doc in item["context"]["sentences"]],
                "answer": {
                    "reasoning": item["cot_answer"]["reasoning"],
                    "final_answer": item["cot_answer"]["final_answer"],
                },
            }
            f.write(json.dumps(raft_item) + "\n")
    logger.info(f"Converted dataset to JSONL format at {output_path}")


def main():
    global ds

    # Get command line arguments
    args = get_args()

    CHUNK_SIZE = args.chunk_size
    NUM_DISTRACT_DOCS = args.distractors

    # Split the document into chunks
    chunks = get_chunks(args.datapath, args.doctype, CHUNK_SIZE)
    logger.info(f"Generated {len(chunks)} chunks from the document")

    ds = None

    num_chunks = len(chunks)

    # Process starting subset of chunks for testing if needed
    # chunks = chunks[:min(10, len(chunks))]
    # num_chunks = len(chunks)

    if not args.fast:
        start = 0
        if os.path.exists("checkpoint.txt"):
            start = int(load_checkpoint("checkpoint.txt"))

        for i in range((start // N) * N, len(chunks)):
            chunk = chunks[i]
            save_checkpoint(i, "checkpoint.txt")

            perc = ceil(i / num_chunks * 100)
            logger.info(f"Processing chunk {i + 1}/{num_chunks} ({perc}%)")
            add_chunk_to_dataset(
                chunks, chunk, args.doctype, args.questions, NUM_DISTRACT_DOCS, args.p, args.qg_model, args.cot_model
            )

            if (i + 1) % N == 0:
                checkpoint_path = f"{args.output}-checkpoints-{i}"
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                ds.save_to_disk(checkpoint_path)
                ds = None

        if ds:
            ds.save_to_disk(f"{args.output}-checkpoints-last")

        ds_list = []
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."

        for filename in os.listdir(output_dir):
            if "-checkpoints-" in filename:
                checkpoint_path = f"{output_dir}/{filename}"
                if os.path.isdir(checkpoint_path):
                    logger.info(f"Loading checkpoint from {checkpoint_path}")
                    for f in os.listdir(checkpoint_path):
                        if f.endswith(".arrow"):
                            ds_list.append(Dataset.from_file(f"{checkpoint_path}/{f}"))

        if ds_list:
            ds = concatenate_datasets(ds_list)
            logger.info(f"Combined {len(ds_list)} checkpoint datasets")
    else:
        for i, chunk in enumerate(chunks):
            perc = ceil(i / num_chunks * 100)
            logger.info(f"Processing chunk {i + 1}/{num_chunks} ({perc}%)")
            add_chunk_to_dataset(
                chunks, chunk, args.doctype, args.questions, NUM_DISTRACT_DOCS, args.p, args.qg_model, args.cot_model
            )

    # Save the final dataset
    if ds:
        logger.info(f"Saving final dataset to {args.output}")
        ds.save_to_disk(args.output)

        # Convert to JSONL format if requested
        if args.output_type.lower() == "jsonl":
            jsonl_path = f"{args.output}.jsonl"
            logger.info(f"Converting dataset to JSONL format at {jsonl_path}")
            convert_to_jsonl(ds, jsonl_path)

    # Clean up checkpoints
    if not args.fast:
        if os.path.exists("checkpoint.txt"):
            os.remove("checkpoint.txt")

        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
        for filename in os.listdir(output_dir):
            if "-checkpoints-" in filename:
                checkpoint_path = f"{output_dir}/{filename}"
                if os.path.isdir(checkpoint_path):
                    logger.info(f"Removing checkpoint directory {checkpoint_path}")
                    shutil.rmtree(checkpoint_path)

    logger.info("RAFT data generation completed successfully!")


if __name__ == "__main__":
    logger.info("Starting the RAFT data generation script...")
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        import traceback

        logger.error(traceback.format_exc())
