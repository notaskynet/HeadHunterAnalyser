from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import torch
import re
import nltk

sentance_model = SentenceTransformer('/app/weights/sbert')
paraphrase_tokenizer = AutoTokenizer.from_pretrained("/app/weights/tokenizer")
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("/app/weights/rut5")

if torch.cuda.is_available():
    paraphrase_model.cuda()


def clean_text(text: str) -> str:
    text = re.sub(r'<[^<]+?>', '', text)
    text = re.sub(r'http\S+', '', text) 
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def process_output(text: str) -> str:
    if text[-1] == ',':
        text = text[:-1]
    if len(text.split(' ')[-1]) < 3:
        text = " ".join(text.split(' ')[:-1])
    return text

def paraphrase_base(text, beams=5, grams=4, do_sample=False, max_new_tokens=500):
    x = paraphrase_tokenizer(text, return_tensors='pt', padding=True).to(paraphrase_model.device)
    out = paraphrase_model.generate(
        input_ids=x['input_ids'],
        attention_mask=x['attention_mask'],
        encoder_no_repeat_ngram_size=grams,
        do_sample=do_sample,
        num_beams=beams,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=4
    )
    return paraphrase_tokenizer.decode(out[0], skip_special_tokens=True)

def clusterize(data: pd.Series, sample_size: int = None, n_clusters: int = 5):
    sentences = data.dropna().astype(str).apply(clean_text).to_list()

    if sample_size is not None:
        sentences = np.random.choice(sentences, sample_size).tolist()
    
    embeddings = sentance_model.encode(sentences, convert_to_tensor=True).cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(embeddings)
    clusters = kmeans.labels_

    cluster_centers = kmeans.cluster_centers_
    summarized_sentences = []
    for i in range(n_clusters):
        cluster_points = np.where(clusters == i)[0]
        cluster_embeddings = embeddings[cluster_points]
        center_embedding = cluster_centers[i]
        distances = np.linalg.norm(cluster_embeddings - center_embedding, axis=1)
        representative_index = cluster_points[np.argmin(distances)]
        summarized_sentences.append(str(sentences[representative_index]))

    return summarized_sentences

def summarize(data: pd.Series, sample_size: int = None, n_clusters: int = 5):
    summarized_sentences = clusterize(data, sample_size, n_clusters)
    summary = []

    for sentence in summarized_sentences:
        if isinstance(sentence, str):
            text = paraphrase_base(sentence, beams=3, grams=7, do_sample=True, max_new_tokens=1000)
            text = process_output(text)
            summary.append(text)
        else:
            print(f"Non-string input encountered: {sentence}, skipping.")
    return summary
