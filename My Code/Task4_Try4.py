import os
import re
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load queries
def load_queries(query_file_path):
    queries = {}
    with open(query_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        raw_queries = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
        for raw_query in raw_queries:
            number = re.search(r'<num> Number: (R\d+)', raw_query).group(1)
            title = re.search(r'<title>(.*?)\n', raw_query).group(1).strip()
            narrative = re.search(r'<narr>(.*?)\n', raw_query).group(1).strip()
            description = re.search(r'<desc>(.*?)\n', raw_query).group(1).strip()
            queries[number] = process_text(title + narrative + description)
            print(queries)
    return queries

# Process text by tokenizing, lowercasing, removing stopwords, and stemming
def process_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

# Load documents from directory
def load_documents(directory_path):
    documents = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf8') as file:
            doc_id = filename.split('.')[0]
            documents[doc_id] = process_text(file.read())
    return documents

# Calculate BM25 scores
def calculate_bm25(N, avgdl, documents, queries, df):
    k1 = 1.2
    k2 = 500
    b = 0.75
    scores = {query_id: {} for query_id in queries}
    for query_id, query in queries.items():
        for doc_id, doc in documents.items():
            score = 0
            dl = len(doc)
            for word in set(query):
                if word in doc:
                    n = df.get(word, 0)
                    f = doc.count(word)
                    qf = query.count(word)
                    K = k1 * ((1 - b) + b * (dl / avgdl))
                    idf = math.log((N - n + 0.5) / (n + 0.5), 10)
                    term_score = idf * ((f * (k1 + 1)) / (f + K)) * ((qf * (k2 + 1)) / (qf + k2))
                    score += term_score
            scores[query_id][doc_id] = score
    return scores

# Calculate JM scores
def calculate_jm_scores(queries, documents, corpus_frequency, corpus_length):
    lambda_param = 0.4
    scores = defaultdict(dict)
    for query_id, query in queries.items():
        for doc_id, doc_tokens in documents.items():
            doc_length = len(doc_tokens)
            score = 0
            for term in query:
                doc_term_freq = doc_tokens.count(term)
                corpus_term_freq = corpus_frequency[term]
                p_td = (1 - lambda_param) * (doc_term_freq / doc_length) if doc_length > 0 else 0
                p_tc = lambda_param * (corpus_term_freq / corpus_length) if corpus_length > 0 else 0
                term_score = p_td + p_tc
                if term_score > 0:
                    score += math.log(term_score)
            scores[query_id][doc_id] = score
    return scores

# Save scores to output folder
def save_scores(scores, output_folder, model_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for query_id, doc_scores in scores.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        output_file_path = os.path.join(output_folder, f"{model_name}_{query_id}Ranking.dat")
        with open(output_file_path, 'w') as file:
            for doc_id, score in sorted_docs:
                file.write(f"{doc_id} {score}\n")
        print(f"Saved {model_name} scores for {query_id} to {output_file_path}")  # Debugging statement

# Main execution flow
query_file_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\the50Queries.txt'
base_data_directory = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\Data_Collection-1\\Data_Collection'
output_folder = 'RankingOutputsTask4_Try4'

queries = load_queries(query_file_path)
print("Loaded Queries:", queries)  # Debugging statement

all_bm25_scores = {}
all_jm_scores = {}

for i in range(101, 151):
    print(f"Processing Data_C{i}")  # Debugging statement to indicate which collection is being processed
    data_directory = os.path.join(base_data_directory, f"Data_C{i}")
    documents = load_documents(data_directory)
    print(f"Loaded Documents for Data_C{i}: {len(documents)} documents")  # Debugging statement
    N = len(documents)
    avgdl = sum(len(doc) for doc in documents.values()) / N
    df = {}
    for doc in documents.values():
        for word in set(doc):
            df[word] = df.get(word, 0) + 1

    bm25_scores = calculate_bm25(N, avgdl, documents, queries, df)
    print(f"BM25 Scores for Data_C{i}: {bm25_scores}")  # Debugging statement
    corpus_len = sum(len(doc) for doc in documents.values())
    corpus_frequency = Counter(token for tokens in documents.values() for token in tokens)
    jm_scores = calculate_jm_scores(queries, documents, corpus_frequency, corpus_len)
    print(f"JM_LM Scores for Data_C{i}: {jm_scores}")  # Debugging statement

    all_bm25_scores.update(bm25_scores)
    all_jm_scores.update(jm_scores)

save_scores(all_bm25_scores, output_folder, 'BM25')
print(f"Saved BM25 scores to {output_folder}")  # Debugging statement
save_scores(all_jm_scores, output_folder, 'JM_LM')
print(f"Saved JM_LM scores to {output_folder}")  # Debugging statement

print("Task 4 completed: Scores for BM25 and JM_LM saved.")
