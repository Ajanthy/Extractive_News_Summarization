import collections
import math

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

def readfile(inputfilename):
    txtFile = open(inputfilename, "rt")
    text = txtFile.read()

    sent_tokens = sent_tokenize(text)
   # cleaning the sentences

    mylineslower = []
    mylines = []
    for sent_i in range(len(sent_tokens)):
        sent = sent_tokens[sent_i]
        sentlower = sent.lower()
        mylineslower.append(sentlower)
        mylines.append(sent)
    return mylineslower, mylines

def title_score(mylineslower):
    lemmatizer = WordNetLemmatizer()
    titletokens = word_tokenize(mylineslower[0])
    lemmatized_titletokens = [lemmatizer.lemmatize(titletoken, pos='v') for titletoken in titletokens]
    tiscore = []

    for line in mylineslower:
        titlescore = 0
        linetokens = word_tokenize(line)
        lemmatized_tokens = [lemmatizer.lemmatize(linetoken, pos='v') for linetoken in linetokens]
        for tokens in lemmatized_tokens:
            for ttokens in lemmatized_titletokens:
                if tokens == ttokens:
                    titlescore += 1
        tiscore.append(titlescore / len(titletokens))
    return tiscore


def sent_length_score(mylineslower):
    lenscore = []
    maxlengthline = len(max(mylineslower, key=len))
    for line in mylineslower:
        lenscore.append(len(line) / maxlengthline)
    return lenscore


def sent_loc_score(mylines):
    count = len(mylines)
    sent_location_score_matrix = []
    for i in range(count):
        if i == 0 or i == count - 1:
            sent_location_score_matrix.append(1)
        else:
            sent_location_score_matrix.append(0)
    return sent_location_score_matrix


def create_dict(mylines, score):
    dictionary = dict(zip(score, mylines))
    sorted_dictionary = collections.OrderedDict(sorted(dictionary.items(), reverse=True))
    return sorted_dictionary

def generate_summary_feature(inputfilename, outputfilename):
    lowersentences, sentence = readfile(inputfilename)
    scoreWithTitle = title_score(lowersentences)
    scoreWithLength = sent_length_score(lowersentences)
    scoreWithLoc = sent_loc_score(lowersentences)

    newlist = [x + y + z for x, y, z in zip(scoreWithLoc, scoreWithTitle, scoreWithLength)]
    ranked_sentences = dict(create_dict(sentence, newlist))
    summarize_text = []
    for i in range(math.ceil(len(ranked_sentences) * 0.4)):
        if i == 0: #skipping this because of title
            continue
        summarize_text.append("".join(list(ranked_sentences.values())[i]))
    summarize_text = ' '.join(map(str, summarize_text))
    print("Summary features: ",  summarize_text)
    outF = open(outputfilename, "w")
    outF.write(summarize_text)

# generate_summary_feature('politicaltext.txt', 'sampleFeatures.txt')
