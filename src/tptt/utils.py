def format_instruction(sample):
    return {
        "text": f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"
    }


def tokenize(sample):
    tokens = tokenizer(
        sample["text"], truncation=True, max_length=256, padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens
