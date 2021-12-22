from nltk import sent_tokenize
from rouge import Rouge

rouge = Rouge()

def read_article(text):
    sentences = sent_tokenize(text)
    return sentences

def evaluate(genfile, reffile):
    textGenFile = open(genfile, "rt")
    textreffile = open(reffile, "rt")
    reference = textreffile.read()
    hypotheses = textGenFile.read()
    scores = rouge.get_scores(hypotheses, reference)
    print("Score for ", genfile, " : ", scores)
    return scores

evaluate('samples/samplekmeans.txt', 'samples/politicalsummary.txt')