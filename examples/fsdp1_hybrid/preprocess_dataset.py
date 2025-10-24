from datasets import load_dataset
import argparse
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zai-org/LongAlign-10k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--random_length", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    return parser.parse_args()


args = get_args()

dataset = (
    load_dataset(args.dataset, split=args.split).shuffle(seed=42).select(range(args.num_samples))
)  # Small subset for quick training
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)


def tokenize_function(examples):
    # Combine question and context for language modeling
    if args.dataset == "zai-org/LongAlign-10k":
        messages = examples["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    elif args.dataset == "SWE-bench/SWE-smith-trajectories":
        import json

        # TODO: Theoretically this should apply chat template, but it's pretraining anyways.
        text = examples["messages"]
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")

    import random

    tokenized = tokenizer(
        text,
        padding=False,
        # max_length=random.randint(1, 16),
    )

    if args.random_length is not None:
        l = random.randint(1, args.random_length)
        while len(tokenized["input_ids"]) < l:
            tokenized["input_ids"] = tokenized["input_ids"][:-1] + tokenized["input_ids"][:-1]
            tokenized["attention_mask"] = (
                tokenized["attention_mask"][:-1] + tokenized["attention_mask"][:-1]
            )
        tokenized["input_ids"] = tokenized["input_ids"][:l]
        tokenized["attention_mask"] = tokenized["attention_mask"][:l]
    if args.max_length is not None:
        tokenized["input_ids"] = tokenized["input_ids"][: args.max_length]
        tokenized["attention_mask"] = tokenized["attention_mask"][: args.max_length]
    # print(text)
    # print(tokenized)
    print(len(tokenized["input_ids"]))
    # For causal language modeling, labels are the same as input_ids
    tokenized["input_ids"] = tokenized["input_ids"]
    tokenized["attention_mask"] = tokenized["attention_mask"]
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized


# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names).shuffle(
    seed=42
)

tokenized_dataset.save_to_disk(args.output)
