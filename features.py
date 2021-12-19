import collections
import math
import nltk
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
WORD = re.compile(r'\w+')

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
        if i == 1 or i == count - 1:
            sent_location_score_matrix.append(1)
        else:
            sent_location_score_matrix.append(0)
    return sent_location_score_matrix

def tokenized_sentences(mylines):
    tokenized_line =[]
    for line in mylines:
        tokenized_line.append(line.split(" "))
    return tokenized_line

def uppercase_availability(mylines):
    upper_case = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    scores = []
    for sentence in tokenized_sentences(mylines):
        score = 0
        for word in sentence:
            if word[0] in upper_case:
                score = score + 1
        scores.append(1.0 * score / len(sentence))
    return scores

def tfisf(mylineslower):
    scores = []
    for sentence in tokenized_sentences(mylineslower):
        counts = collections.Counter(sentence)
        score = 0
        for word in counts.keys():
            count_word = 0
            for sent in tokenized_sentences(mylineslower):
                for w in sent:
                    if word == w:
                        count_word += 1
            score = score + counts[word] * math.log(count_word)
        scores.append(score / len(sentence))
    return scores

def similar(tokens_a, tokens_b):
    # Using Jaccard similarity to calculate if two sentences are similar
    ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
    return ratio


def similarityScores(mylineslower):
    scores = []
    for sentence in tokenized_sentences(mylineslower):
        score = 0
        for sen in tokenized_sentences(mylineslower):
            if sen != sentence:
                score += similar(sentence, sen)
        scores.append(score)
    return scores

def posTagger(mylineslower):
    tagged = []
    for sentence in tokenized_sentences(mylineslower):
        tag = nltk.pos_tag(sentence)
        tagged.append(tag)
    return tagged # returns an list of list with words and tags of each sentences

def properNounScores(tagged):
    scores = []
    for i in range(len(tagged)):
        score = 0
        for j in range(len(tagged[i])):
            if (tagged[i][j][1] == 'NNP' or tagged[i][j][1] == 'NNPS'): #seperate propernouns (Proper Noun Singular and Proper Noun Plural)
                score += 1
        scores.append(score / float(len(tagged[i])))
    return scores

def text_to_vector(text):
    words = WORD.findall(text)
    return collections.Counter(words)

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def centroidSimilarity(sentences, tfIsfScore):
    centroidIndex = tfIsfScore.index(max(tfIsfScore))
    scores = []
    for sentence in sentences:
        vec1 = text_to_vector(sentences[centroidIndex])
        vec2 = text_to_vector(sentence)

        score = get_cosine(vec1, vec2)
        scores.append(score)
    return scores

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def numericToken(tokenized_sentences):
    scores = []
    for sentence in tokenized_sentences:
        score = 0
        for word in sentence:
            if is_number(word):
                score += 1
        scores.append(score / float(len(sentence)))
    return scores

def create_dict(mylines, score):
    mylines = mylines[1:]
    score = score[1:]
    dictionary = dict(zip(score, mylines))
    sorted_dictionary = collections.OrderedDict(sorted(dictionary.items(), reverse=True))
    return sorted_dictionary

def generate_summary_feature(inputfilename, outputfilename):
    lowersentences, sentence = readfile(inputfilename)
    scoreWithTitle = title_score(lowersentences)
    scoreWithLength = sent_length_score(lowersentences)
    scoreWithLoc = sent_loc_score(lowersentences)
    scorewithuppercase = uppercase_availability(sentence)
    scoretfisf = tfisf(lowersentences)
    scorejaccardsimilarity = similarityScores(lowersentences)
    scorePropernoun = properNounScores(posTagger(lowersentences))
    scorecentroidsimilarity = centroidSimilarity(sentence, scoretfisf)
    scorenumerictoken = numericToken(tokenized_sentences(sentence))

    newlist = [x + y + z + a +b + c + d + e + f  for x, y, z, a, b, c, d, e,f in zip(scoreWithLoc, scoreWithTitle, scoreWithLength, scorewithuppercase,scoretfisf,scorejaccardsimilarity,scorePropernoun, scorenumerictoken,scorecentroidsimilarity)]
    ranked_sentences = dict(create_dict(sentence, newlist))
    summarize_text = []
    for i in range(math.ceil(len(ranked_sentences) * 0.4)):
        summarize_text.append("".join(list(ranked_sentences.values())[i]))
    summarize_text = ' '.join(map(str, summarize_text))
    print("Summary features: ",  summarize_text)
    outF = open(outputfilename, "w")
    outF.write(summarize_text)

# generate_summary_feature('politicaltext.txt', 'sampleFeatures.txt')
