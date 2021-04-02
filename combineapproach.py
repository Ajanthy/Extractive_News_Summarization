import os
import csv

from nltk import sent_tokenize
from trycosinewithpagerank import generate_summary_textrank
from features import generate_summary_feature
from kmeans import generate_summary_kmeans
from rougeurdu import evaluate

filepath = '../BBC News Summary/News Articles/politics'
summarypath = '../BBC News Summary/Summaries/politics'
final_file  = open('final_file.csv', mode='w')
final_evaluation = csv.writer(final_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
final_evaluation.writerow(['Evaluation for Final combined Summary'])
final_evaluation.writerow(['File Name', 'ROUGE', 'Precision', 'Recall', 'F-measure'])

def readFile_token(outputfilename):
    textFile = open(outputfilename, "rt")
    text = textFile.read()
    textFile.close()
    sent_tokens = sent_tokenize(text)
    return sent_tokens

for f in os.listdir(filepath):
    print(f)
    input = filepath+'/'+f
# input = 'politicaltext.txt'
    generate_summary_feature(input, 'sampleFeatures.txt')
    generate_summary_kmeans(input, 'samplekmeans.txt')
    generate_summary_textrank(input, 'samplecosine.txt')

    sent_tokens_cosine = readFile_token('samplecosine.txt')
    sent_tokens_kmeans = readFile_token('samplekmeans.txt')
    sent_tokens_features = readFile_token('sampleFeatures.txt')

    reffile = summarypath+'/'+f
    z = set(sent_tokens_cosine).union(sent_tokens_features)
    final = z.union(sent_tokens_kmeans)
    finalsummary = " ".join(final)
    outF = open('finaloutput.txt', "w")
    outF.write(finalsummary)
    outF.close()
    print("Final Summary:", finalsummary)

    scores = evaluate('finaloutput.txt', reffile)
    print(scores)
    scores[0] = dict(scores[0])
    for key in scores[0]:
        final_evaluation.writerow([f, key, scores[0][key]['p'], scores[0][key]['r'], scores[0][key]['f']])