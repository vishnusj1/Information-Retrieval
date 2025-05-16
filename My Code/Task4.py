import os
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Paths
base_data_directory = 'C:/Users/samin/Desktop/IFN647/Assignment 2/Data_Collection-1/Data_Collection'
query_file_path = 'C:/Users/samin/Desktop/IFN647/Assignment 2/the50Queries.txt'
output_folder = 'C:/Users/samin/Desktop/IFN647/Assignment 2/My Code/RankingOutputs'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load and process stopwords
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

stop_words_file_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\common-english-words.txt'
stop_words = load_stop_words(stop_words_file_path)

# Text processing function
def process_text(text, stop_words):
    tokens = word_tokenize(text.lower())
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalnum()]

# Loading documents
def load_documents(directory_path):
    documents = {}
    corpus_length = 0
    document_lengths = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf8') as file:
            tokens = process_text(file.read().strip(), stop_words)
            documents[filename] = tokens
            doc_length = len(tokens)
            document_lengths[filename] = doc_length
            corpus_length += doc_length
    return documents, document_lengths, corpus_length

def build_corpus_frequency(documents):
    return Counter(token for tokens in documents.values() for token in tokens)

# Load queries
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

def load_documents_bm25(directory_path):
    documents = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf8') as file:
            text = file.read().strip()
            tokens = process_text(text, stop_words)
            documents[filename] = tokens
    return documents
# Calculation functions for BM25 and Jelinek-Mercer
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
                score += p_td + p_tc
            scores[query_id][doc_id] = score
    return scores

# Saving scores
def save_scores(scores, output_folder, prefix):
    for query_id, doc_scores in scores.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        output_file_path = os.path.join(output_folder, f"{prefix}_{query_id}Ranking.dat")
        with open(output_file_path, 'w') as file:
            for doc_id, score in sorted_docs:
                file.write(f"{doc_id}\t{score}\n")

# Main execution flow
queries = load_queries(query_file_path, stop_words)
all_scores_bm25 = {}
all_scores_jm = {}

for i in range(101, 151):
    data_directory = os.path.join(base_data_directory, f"Data_C{i}")
    documents, document_lengths, corpus_length = load_documents(data_directory)
    freq = build_corpus_frequency(documents)

    #variables needed for bm25-
    documents_bm25 = load_documents_bm25(data_directory)
    N = len(documents_bm25)
    avgdl = sum(len(doc) for doc in documents.values()) / N
    df = {}


    # BM25
    # df = {word: sum(1 for d in documents if word in d) for word in freq}
    # scores_bm25 = calculate_bm25(len(documents), sum(document_lengths.values()) / len(document_lengths), documents, queries, df)
    # all_scores_bm25.update(scores_bm25)

    for doc in documents.values():
        for word in set(doc):
            df[word] = df.get(word, 0) + 1
    scores = calculate_bm25(N, avgdl, documents_bm25, queries, df)
    all_scores_bm25.update(scores)

    # JM
    scores_jm = calculate_jm_scores(queries, documents, freq, corpus_length)
    all_scores_jm.update(scores_jm)

save_scores(all_scores_bm25, output_folder, "BM25")
save_scores(all_scores_jm, output_folder, "JM")
