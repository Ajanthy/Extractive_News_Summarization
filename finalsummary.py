import os
import csv

from nltk import sent_tokenize
from graph import generate_summary_textrank
from features import generate_summary_feature
from kmeans import generate_summary_kmeans
from rougeurdu import evaluate

filepath = '../BBC News Summary/News Articles/politics'
summarypath = '../BBC News Summary/Summaries/politics'
final_file  = open('csvs/final_file.csv', mode='w')
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
    generate_summary_feature(input, 'samples/sampleFeatures.txt')
    generate_summary_kmeans(input, 'samples/samplekmeans.txt')
    generate_summary_textrank(input, 'samples/samplecosine.txt')

    sent_tokens_cosine = readFile_token('samples/samplecosine.txt')
    sent_tokens_kmeans = readFile_token('samples/samplekmeans.txt')
    sent_tokens_features = readFile_token('samples/sampleFeatures.txt')

    reffile = summarypath+'/'+f
    z = set(sent_tokens_cosine).union(sent_tokens_features)
    # final = set(sent_tokens_kmeans).union(sent_tokens_cosine)
    final = z.union(sent_tokens_kmeans)

    finalsummary = " ".join(final)
    outF = open('samples/finaloutput.txt', "w")
    outF.write(finalsummary)
    outF.close()
    print("Final Summary:", finalsummary)

    scores = evaluate('samples/finaloutput.txt', reffile)
    print(scores)
    scores[0] = dict(scores[0])
    for key in scores[0]:
        final_evaluation.writerow([f, key, scores[0][key]['p'], scores[0][key]['r'], scores[0][key]['f']])