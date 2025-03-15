import re
import numpy as np
import pandas as pd
from transformers import AutoModel, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


embedding_model = AutoModel.from_pretrained(
    "/app/models/jina-embeddings-v3", trust_remote_code=True
)
pipe = pipeline(
    "text-generation",
    model="/app/models/Vikhr-Llama-3.2-1B-Instruct",
    max_new_tokens=1024,
    temperature=0.3,
    repetition_penalty=1.1,
)
llm = HuggingFacePipeline(pipeline=pipe)


def clean_text(text: str) -> str:
    text = re.sub(r"<[^<]+?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def process_output(text: str) -> str:
    if text[-1] == ",":
        text = text[:-1]
    if len(text.split(" ")[-1]) < 3:
        text = " ".join(text.split(" ")[:-1])
    return text


def generate_embeddings(texts: list[str]) -> np.ndarray:
    return embedding_model.encode(texts, task="text-matching")


def clusterize(data: pd.Series, sample_size: int = None, n_clusters: int = 5):
    sentences = data.dropna().astype(str).apply(clean_text).to_list()

    if sample_size is not None:
        sentences = np.random.choice(sentences, sample_size).tolist()

    embeddings = generate_embeddings(sentences)

    from sklearn.cluster import KMeans

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


def paraphrase(text: str) -> str:
    prompt = PromptTemplate.from_template(
        """Rewrite the following sentence with the same meaning but different wording:
        
        {text}
        
        Paraphrased:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"text": text})
    return process_output(result["text"])


def summarize(data: pd.Series, sample_size: int = None, n_clusters: int = 5):
    summarized_sentences = clusterize(data, sample_size, n_clusters)
    summary = [paraphrase(sentence) for sentence in summarized_sentences]
    return summary
