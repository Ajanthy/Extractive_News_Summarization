import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance  # Measure Similarity between sentences
import numpy as np
import networkx as nx  # Creating and manipulating graph

# text = """Claxton hunting first major medal
#
# British hurdler Sarah Claxton is confident she can win her first major medal at next month's European Indoor Championships in Madrid.
#
# The 25-year-old has already smashed the British record over 60m hurdles twice this season, setting a new mark of 7.96 seconds to win the AAAs title. "I am quite confident," said Claxton. "But I take each race as it comes. "As long as I keep up my training but not do too much I think there is a chance of a medal." Claxton has won the national 60m hurdles title for the past three years but has struggled to translate her domestic success to the international stage. Now, the Scotland-born athlete owns the equal fifth-fastest time in the world this year. And at last week's Birmingham Grand Prix, Claxton left European medal favourite Russian Irina Shevchenko trailing in sixth spot.
#
# For the first time, Claxton has only been preparing for a campaign over the hurdles - which could explain her leap in form. In previous seasons, the 25-year-old also contested the long jump but since moving from Colchester to London she has re-focused her attentions. Claxton will see if her new training regime pays dividends at the European Indoors which take place on 5-6 March.
# """

# Read the text and tokenize into sentences
def read_article(text):
    sentences = sent_tokenize(text)
    return sentences

# Create vectors and calculate cosine similarity b/w two sentences
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


# Create similarity matrix among all sentences
def build_similarity_matrix(sentences, stop_words):
    # create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


# Generate and return text summary
def generate_summary_textrank(inputfilename, outputfilename, top_n = 5):
    # Read text from file
    txtFile = open(inputfilename, "r")
    title = txtFile.readline()
    text = txtFile.read()
    txtFile.close()

    stop_words = stopwords.words('english')
    summarize_text = []

    # Step1: read text and tokenize
    sentences = read_article(text)

    # Steo2: generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # Step3: Rank sentences in similarirty matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step4: sort the rank and place top sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Step 5: get the top n number of sentences based on rank
    for i in range(top_n):
        summarize_text.append("".join(ranked_sentences[i][1]))

    # Step 6 : outpur the summarized version
    summarize_text = " ".join(summarize_text)
    print("Summarize Text textrank: \n", summarize_text)
    outF = open(outputfilename, "w")
    outF.write(summarize_text)
    outF.close()

# generate_summary_textrank('politicaltext.txt', 'samplecosine.txt')
