import os
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Define file paths
# document_path = 'C:/Users/pallavi/PycharmProjects/Assignment_Project2/Data_Collection-1/Data_Collection/'
# query_file_path = 'C:/Users/pallavi/PycharmProjects/Assignment_Project2/the50Queries.txt'
# output_path = 'C:/Users/pallavi/PycharmProjects/Assignment_Project2/333Outputs-Task3/'
# stop_words_file_path = 'C:/Users/pallavi/PycharmProjects/Assignment_Project2/common-english-words.txt'

query_file_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\the50Queries.txt'
document_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\Data_Collection-1\\Data_Collection'
output_path = 'Outputs-Task3 - Pallavi'
stop_words_file_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\common-english-words.txt'

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load stop words
def load_stop_words(file_path):
    stop_words = set(stopwords.words('english'))
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            line = file.readline()
            custom_stop_words = line.strip().split(',')
            stop_words.update(word.strip() for word in custom_stop_words)
    except IOError:
        print("Error opening or reading input file:", file_path)
    return stop_words

stop_words = load_stop_words(stop_words_file_path)

# Text processing function
def process_text(text, stop_words):
    tokens = word_tokenize(text.lower())
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalnum()]

# Load documents and calculate document frequency for terms
def load_documents_and_build_df(directory_path):
    documents = {}
    df = defaultdict(int)  # Document frequency of each term
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf8') as file:
                    text = file.read().strip()
                    tokens = process_text(text, stop_words)
                    documents[filename] = tokens
                    unique_terms = set(tokens)
                    for term in unique_terms:
                        df[term] += 1
            except IOError:
                print(f"Error opening or reading document file: {file_path}")
    return documents, df

# Load queries
def load_queries(query_file_path):
    queries = {}
    with open(query_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        raw_queries = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
        for raw_query in raw_queries:
            number = re.search(r'<num> Number: (R\d+)', raw_query).group(1)
            title = re.search(r'<title>(.*?)\n', raw_query).group(1).strip()
            queries[number] = process_text(title, stop_words)
    return queries

# Calculate BM25, using pre-built document frequency dictionary
def calculate_bm25(documents, queries, df, N):
    avgdl = sum(len(doc) for doc in documents.values()) / N
    scores = defaultdict(dict)
    for query_id, query in queries.items():
        for doc_id, doc in documents.items():
            dl = len(doc)
            score = 0
            for term in query:
                f = doc.count(term)
                n = df[term]
                K = 1.2 * ((1 - 0.75) + 0.75 * dl / avgdl)
                idf = math.log((N - n + 0.5) / (n + 0.5)) if n > 0 else 0
                score += idf * f * (1.2 + 1) / (f + K)
            scores[query_id][doc_id] = score
    return scores

# Save scores to files with filtering
def save_scores(scores, output_folder, top_n=50, score_threshold=0.1):
    for query_id, doc_scores in scores.items():
        output_file_path = os.path.join(output_folder, f"My_PRM_{query_id}Ranking.dat")
        valid_scores = {doc_id: score for doc_id, score in doc_scores.items() if score > score_threshold}
        with open(output_file_path, 'w') as file:
            for doc_id, score in sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
                file.write(f"{doc_id}\t{score}\n")

# Main execution flow
queries = load_queries(query_file_path)
documents, df = load_documents_and_build_df(document_path)
N = len(documents)  # Total number of documents
initial_scores = calculate_bm25(documents, queries, df, N)
save_scores(initial_scores, output_path, top_n=50, score_threshold=0.1)
