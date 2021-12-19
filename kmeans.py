import math

import nltk
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from scipy.spatial import distance
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# sentence tokenization

from nltk.tokenize import sent_tokenize

def generate_summary_kmeans(inputfilename, outputfilename):

    txtFile = open(inputfilename, "rt")
    title = txtFile.readline()
    text = txtFile.read()

    sent_tokens = sent_tokenize(text)

    # cleaning the sentences

    sent_tokens_list = []
    for cluster_i in range(len(sent_tokens)):
        Punctuation_Tokenizer = nltk.RegexpTokenizer(r"\w+")  # Matches Alphanumeric characters
        sent = re.sub('[^a-zA-Z]', " ", sent_tokens[cluster_i])  # Only the word without numbers are extracted to variable sent
        sent = sent.lower()
        sent = Punctuation_Tokenizer.tokenize(sent)
        lemmatizer = WordNetLemmatizer() # Lemmatizing
        sent = [lemmatizer.lemmatize(token, pos='v') for token in sent]
        sent = ' '.join([i for i in sent if i not in stopwords.words('english')])  # stop words removal
        sent_tokens_list.append(sent)
    # creating word vectors
    max_number_of_unique_words = 500
    word_tokens = [sentTokens.split() for sentTokens in sent_tokens_list]
    model = Word2Vec(word_tokens, min_count=1, size=max_number_of_unique_words)

    # creating sentence vectors
    sen_vector = []
    for sent_i in sent_tokens_list:  # sentence list
        count = 0
        for word_i in sent_i.split():
            count += model.wv[word_i]
        count = count / len(sent_i.split())
        sen_vector.append(count)
    # performing k-means
    n_clusters = int(math.ceil(len(sent_tokens) * 0.1))
    kMeans = KMeans(n_clusters)
    y_kMeans = kMeans.fit_predict(sen_vector)  # get the cluster center and predict the cluster index for each sentence
    # finding and printing the nearest sentence vector from cluster centroid

    my_list = []
    for cluster_i in range(n_clusters):  # In the range of clusters
        my_dict = {}  # dict is null
        for sent_i in range(len(y_kMeans)):  # In the range of cluster index where it belongs
            if y_kMeans[sent_i] == cluster_i:  # if both cluster index and sentence's index matches
                my_dict[sent_i] = distance.cosine(kMeans.cluster_centers_[cluster_i], sen_vector[sent_i])  # get the cosine of that vector
        my_list.append(min(my_dict, key=my_dict.get))  # get the sentence which has minimum distance in that cluster

    #Print the summary
    summarize_text = []
    for selected_sent_i in sorted(my_list):
        summarize_text.append(sent_tokens[selected_sent_i])
    summarize_text = ' '.join(map(str, summarize_text))
    outF = open(outputfilename, "w")
    outF.write(summarize_text)
    # print("Summary kmeans : ", summarize_text)

generate_summary_kmeans('samples/politicaltext.txt', 'samples/samplekmeans.txt')