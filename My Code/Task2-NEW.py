# import os
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from collections import defaultdict, Counter
#
# # Ensure necessary NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

# def process_text(text, stop_words):
#     """
#     Tokenize, lower-case, remove stopwords, and stem the text.
#     """
#     tokens = word_tokenize(text.lower())
#     stemmer = PorterStemmer()
#     return [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalnum()]

def load_stop_words(file_path):
    """
    Load custom stop words from a file, adding them to the default English stop words.
    """
    stop_words = set(stopwords.words('english'))
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words.update(word.strip() for word in file.read().strip().split(','))
    return stop_words

# def load_documents(directory_path):
#     #save documents in a dictionary
#     documents = {}
#     corpus_length = 0
#     document_lengths = {}  # Dictionary to store individual document lengths
#     for filename in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, filename)
#         with open(file_path, 'r', encoding='utf8') as file:
#             text = file.read().strip()
#             tokens = process_text(text, stop_words)
#             documents[filename] = tokens
#             #print("document", documents[filename])
#             doc_length = len(tokens)  # Length of this particular document
#             #print("doc length", doc_length)
#             document_lengths[filename] = doc_length  # Store individual document length
#             corpus_length += doc_length
#             #print("corpus length", corpus_length)
#     return documents, document_lengths, corpus_length

# def build_corpus_frequency(documents):
#     """
#     Build frequency distribution of terms across all documents.
#     """
#     return Counter(token for tokens in documents.values() for token in tokens)

# def calculate_jm_scores(queries, documents, corpus_frequency, corpus_length, lambda_param=0.4):
#     """
#     Calculate JM smoothing scores for each document against each query.
#     """
#     scores = defaultdict(dict)
#     for query_id, query_tokens in queries.items():
#         for doc_id, doc_tokens in documents.items():
#             doc_length = len(doc_tokens)
#             score = 0
#             for term in query_tokens:
#                 doc_term_freq = doc_tokens.count(term)
#                 corpus_term_freq = corpus_frequency[term]
#                 p_td = (1 - lambda_param) * (doc_term_freq / doc_length) if doc_length > 0 else 0
#                 p_tc = lambda_param * (corpus_term_freq / corpus_length) if corpus_length > 0 else 0
#                 score += p_td + p_tc
#             scores[query_id][doc_id] = score
#     return scores

# def save_scores(scores, output_folder):
#     """
#     Save scores to files, one for each query.
#     """
#     for query_id, doc_scores in scores.items():
#         sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
#         output_file_path = os.path.join(output_folder, f"JM_LM_{query_id}Ranking.dat")
#         with open(output_file_path, 'w') as file:
#             for doc_id, score in sorted_docs:
#                 file.write(f"{doc_id}\t{score}\n")

# def load_queries(query_file_path, stop_words):
#     """
#     Load queries from a file, process and tokenize each.
#     """
#     queries = {}
#     with open(query_file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#         raw_queries = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
#         for raw_query in raw_queries:
#             number = re.search(r'<num> Number: (\w+)', raw_query).group(1)
#             title = re.search(r'<title>(.*?)\n', raw_query).group(1).strip()
#             description_search = re.search(r'<desc> Description:\s*(.*?)(?=\n<narr>|</Query>)', raw_query, re.DOTALL)
#             narrative_search = re.search(r'<narr> Narrative:\s*(.*?)\n\n', raw_query, re.DOTALL)
#
#             description = description_search.group(1).strip() if description_search else ""
#             narrative = narrative_search.group(1).strip() if narrative_search else ""
#
#             full_query = f"{title} {description} {narrative}"
#             queries[number] = process_text(full_query, stop_words)
#     return queries

# Define file paths and directories
stop_words_file = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\common-english-words.txt'
query_file_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\the50Queries.txt'
output_folder = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\My Code\\Outputs-Task2-New\\'
document_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\Data_Collection-1\\Data_Collection\\'

# Load stop words
stop_words = load_stop_words(stop_words_file)

# Load and process queries
queries = load_queries(query_file_path, stop_words)

# Main processing loop for each query
for query_id in queries:
    data_directory = os.path.join(document_path, f"Data_C{query_id[1:]}")
    documents, document_lengths, corpus_length = load_documents(data_directory)
    freq = build_corpus_frequency(documents)
    scores = calculate_jm_scores({query_id: queries[query_id]}, documents, freq, corpus_length)
    save_scores(scores, output_folder)
