from nltk import sent_tokenize
from rouge import Rouge

rouge = Rouge()

# reffile = open('politicalsummary.txt', "rt")
# # Title = reffile.readline().rstrip()  # Discard first line
# reference = reffile.read()
#
# hypfile = open('sampleFeatures.txt', "rt")
# Title = txtfile.readline().rstrip()  # Discard first line
# hypotheses = hypfile.read()
#
# hypfile = open('finaloutput.txt', "rt")
# # Title = txtfile.readline().rstrip()  # Discard first line
# hypotheses = hypfile.read()

def read_article(text):
    sentences = sent_tokenize(text)
    return sentences

def evaluate(genfile, reffile):
    textGenFile = open(genfile, "rt")
    textreffile = open(reffile, "rt")
    reference = textreffile.read()
    print("No of sentences in reference : ", len(read_article(reference)))
    hypotheses = textGenFile.read()
    scores = rouge.get_scores(hypotheses, reference)
    print("Score for ", genfile, " : ", scores)
    return scores

evaluate('samples/samplekmeans.txt', 'samples/politicalsummary.txt')