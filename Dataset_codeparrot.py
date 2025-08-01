import os
import json
from datasets import load_dataset

# Settings
OUTPUT_DIR = "dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "codeparrot_40k.json")

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[*] Loading ~4% of CodeParrot Clean dataset (aiming for ~40k samples)...")
dataset = load_dataset(
    "codeparrot/codeparrot-clean",
    split="train[:4%]",
    revision="refs/convert/parquet"  # compatibility with dataset structure
)

print(f"[✓] Loaded {len(dataset)} raw samples.")

processed_data = []

print("[*] Preprocessing...")

for sample in dataset:
    code = sample.get("content", "").strip()

    # Filter out short or low-quality code
    if len(code.splitlines()) < 3 or "copyright" in code.lower():
        continue

    prompt = "Write a Python function that accomplishes the following task:\n\n"
    processed_data.append({
        "prompt": prompt,
        "completion": code
    })

print(f"[✓] Filtered and prepared {len(processed_data)} samples.")

print(f"[✓] Saving to {OUTPUT_FILE}...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2)

print("[✅] Dataset creation complete.")
