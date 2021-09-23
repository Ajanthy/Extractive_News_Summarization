import random
from nltk.tokenize import sent_tokenize
import math
from rougeurdu import evaluate

def read_article(text):
    sentences = sent_tokenize(text)
    return sentences

def getcount(sentences):
    n = int(math.ceil(len(sentences) * 0.2))
    print(n)
    return n

def selectsentences(sentences, n):
    randomlist = random.sample(range(1, len(sentences)), n)
    # randomlist = []
    # for i in range(n):
    #     randomlist.append(i)
    print(randomlist)
    return  sorted(randomlist)

def printsummary(randomlist, sentences, outputfilename):
    summarize_text = []
    for i in randomlist:
        summarize_text.append("".join(list(sentences[i])))
    summarize_text = ' '.join(map(str, summarize_text))
    print("Summary Random: ", summarize_text)
    outF = open(outputfilename, "w")
    outF.write(summarize_text)

def generate_summary_random(inputfilename,outputfilename):
    txtFile = open(inputfilename, "r")
    title = txtFile.readline()
    text = txtFile.read()
    txtFile.close()

    randomlist = selectsentences(read_article(text),getcount(read_article(text)))

    printsummary(randomlist, read_article(text), outputfilename)

generate_summary_random('politicaltext.txt', 'results.txt')
evaluate('results.txt', 'politicalsummary.txt')