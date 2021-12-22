from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance  # Measure Similarity between sentences
import numpy as np
import networkx as nx  # Creating sentence similarity graph
import math

# Read the text and tokenize into sentences
def read_article(text):
    sentences = sent_tokenize(text)
    return sentences

# Create vectors and calculate cosine similarity between two sentences
def sentence_similarity(sent1, sent2, stopWords=None):
    if stopWords is None:
        stopWords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopWords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopWords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


# Create similarity matrix among all sentences
def build_similarity_matrix(sentences, stop_words):
    # create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for sent1 in range(len(sentences)):
        for sent2 in range(len(sentences)):
            if sent1 != sent2:
                similarity_matrix[sent1][sent2] = sentence_similarity(sentences[sent1], sentences[sent2], stop_words)

    return similarity_matrix


# Generate and return text summary
def generate_summary_textrank(inputfilename, outputfilename):
    # Read text from file
    txtFile = open(inputfilename, "r")
    title = txtFile.readline()
    text = txtFile.read()
    txtFile.close()

    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1: read text and tokenize
    sentences = read_article(text)

    # Step 2: generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # Step 3: Rank sentences in similarity graph
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4: sort the rank and place top sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Step 5: get the top n number of sentences based on rank
    for i in range(int(math.ceil(len(sentences) * 0.2))):
        summarize_text.append("".join(ranked_sentences[i][1]))

    # Step 6 : output the summarized version
    summarize_text = " ".join(summarize_text)
    print("Summarize Text : \n", summarize_text)
    outF = open(outputfilename, "w")
    outF.write(summarize_text)
    outF.close()

generate_summary_textrank('samples/politicaltext.txt', 'samples/samplecosine.txt')
