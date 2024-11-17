from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


PATH_TO_SAVE = "./app/model/weights"

# Downloading weights for S-BERT

print("Downloading weights for S-Bert...")
sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
sentence_model.save(f"{PATH_TO_SAVE}/sbert")

# Downloading weights for RuT5 Tokenizer
print("Downloading weights for RuT5 Tokenizer...")
paraphrase_tokenizer = AutoTokenizer.from_pretrained("XvKuoMing/t5-rebuilder-v2")
paraphrase_tokenizer.save_pretrained(f"{PATH_TO_SAVE}/tokenizer")

# Downloading weights for RuT5 Model
print("Downloading weights for RuT5 Model...")
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("XvKuoMing/t5-rebuilder-v2")
paraphrase_model.save_pretrained(f"{PATH_TO_SAVE}/rut5")