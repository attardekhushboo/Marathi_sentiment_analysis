from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Path to your fine-tuned checkpoint folder (adjust as needed)
CHECKPOINT_PATH = "checkpoint-12030"

# Load model from checkpoint
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH)

# Load original tokenizer (from Hugging Face model hub)
tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/marathi-sentence-bert-nli")

# Save both model and tokenizer to a clean directory
SAVE_DIR = "mahasentiment_model"

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model and tokenizer saved to '{SAVE_DIR}'")
