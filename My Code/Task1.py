import os
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def process_text(text, stop_words):
    """
    Returns:
    list: A list of stemmed tokens.
    """
    tokens = word_tokenize(text.lower())  # Tokenize the text and convert to lower case.
    stemmer = PorterStemmer()  # Initialize the PorterStemmer.
    # Filter out stopwords and stem the remaining words
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalnum()]
    #print(stemmed_tokens)

    return stemmed_tokens

def load_stop_words(file_path):
    # Start with the default English stop words from NLTK
    stop_words = set(stopwords.words('english'))

    # Open the file and read stop words from it
    with open(file_path, 'r', encoding='utf-8') as file:
        # Since all words are on one line, read the single line
        line = file.readline()
        # Split the line into words based on commas and strip whitespace
        custom_stop_words = line.strip().split(',')
        # Add each word from the file to the stop words set
        stop_words.update(word.strip() for word in custom_stop_words)
    # #had to add this because these are useless to the processing

    custom_stopwords = {'xml', 'newsitem', 'root', 'en', 'titl'}
    stop_words.update(custom_stopwords)

    return stop_words

file_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\common-english-words.txt'  # Replace with the actual file path
stop_words = load_stop_words(file_path)

def load_queries(query_file_path, stop_words):
    queries = {}
    with open(query_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        raw_queries = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
        for raw_query in raw_queries:
            number = re.search(r'<num> Number: (R\d+)', raw_query).group(1)
            title = re.search(r'<title>(.*?)\n', raw_query).group(1).strip()
            description_search = re.search(r'<desc> Description:\s*(.*?)(?=\n<narr>|</Query>)', raw_query, re.DOTALL)
            narrative_search = re.search(r'<narr> Narrative:\s*(.*?)\n\n', raw_query, re.DOTALL)

            description = description_search.group(1).strip() if description_search else ""
            narrative = narrative_search.group(1).strip() if narrative_search else ""

            full_query = f"{title} {description} {narrative}"
            queries[number] = process_text(full_query, stop_words)
        return queries

def load_documents(directory_path):
    documents = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf8') as file:
            text = file.read().strip()
            tokens = process_text(text, stop_words)
            documents[filename] = tokens
            #print(filename)
    return documents

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
                    K = k1 * ((1 - b) + b * dl / avgdl)
                    idf = math.log((N - n + 0.5) / (n + 0.5), 10)
                    term_score = idf * ((f * (k1 + 1)) / (f + K)) * ((qf * (k2 + 1)) / (qf + k2))
                    score += term_score
            scores[query_id][doc_id] = score
    return scores

def save_scores(scores, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for query_id, doc_scores in scores.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        output_file_path = os.path.join(output_folder, f"BM25_{query_id}Ranking.dat")
        with open(output_file_path, 'w') as file:
            for doc_id, score in sorted_docs:
                file.write(f"{doc_id} {score}\n")

# Example usage
query_file_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\the50Queries.txt'
base_data_directory = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\Data_Collection-1\\Data_Collection'
output_folder = 'Outputs-Task1-New'

queries = load_queries(query_file_path, stop_words)
all_scores = {}

for i in range(101, 151):
    data_directory = os.path.join(base_data_directory, f"Data_C{i}")
    print(data_directory)
    documents = load_documents(data_directory)
    N = len(documents)
    #print(N)
    avgdl = sum(len(doc) for doc in documents.values()) / N
    df = {}
    for doc in documents.values():
        for word in set(doc):
            df[word] = df.get(word, 0) + 1
    scores = calculate_bm25(N, avgdl, documents, queries, df)
    print(scores)
    all_scores.update(scores)
    print(all_scores)
save_scores(all_scores, output_folder)

