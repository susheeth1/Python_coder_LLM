# tokenization.py
from tokenizers.implementations import ByteLevelBPETokenizer
import json

def train_tokenizer():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files="dataset/codeparrot_40k.json",
        vocab_size=8000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    tokenizer.save_model("tokenizer")

if __name__ == "__main__":
    train_tokenizer()
