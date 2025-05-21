import json
from tokenizers import ByteLevelBPETokenizer

# train a BPE tokenizer on your corpus of instructions + responses
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["train_texts.txt"], 
    vocab_size=30000, 
    min_frequency=2, 
    special_tokens=["<pad>", "<unk>", "<s_thought>", "<s_answer>"]
)
tokenizer.save_model("tokenizer")

# create a small sample of your multimodal training data
samples = [
    {
        "instruction": "Describe the contents of the image.",
        "response": "A cat sleeping on a red cushion.",
        "image_path": "images/cat1.jpg",
        "safety_label": 0
    },
    {
        "instruction": "What is happening in the scene?",
        "response": "A group of people enjoying a picnic in the park.",
        "image_path": "images/picnic1.jpg",
        "safety_label": 0
    },
    {
        "instruction": "Identify any dangerous objects in the picture.",
        "response": "I see a lit candle near curtains, which could be a fire hazard.",
        "image_path": "images/candle1.jpg",
        "safety_label": 1
    }
]

# (optional) inspect how BPE tokenizes a combined prompt+answer
for ex in samples:
    text = f"<s_thought> {ex['instruction']} <s_answer> {ex['response']}"
    enc = tokenizer.encode(text)
    print(ex["image_path"], enc.ids, "\n")

# write out the JSON file that JSONDataset expects
with open("train.json", "w") as f:
    json.dump(samples, f, indent=2)
