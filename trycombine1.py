import os
import csv

from nltk import sent_tokenize
from graph import generate_summary_textrank
from features import generate_summary_feature
from kmeans import generate_summary_kmeans
from rougeurdu import evaluate

def readFile_token(outputfilename):
    textFile = open(outputfilename, "rt")
    text = textFile.read()
    textFile.close()
    sent_tokens = sent_tokenize(text)
    return sent_tokens

combinefinal_file = open('csvs/combinefinal_file.csv', mode='w')
combine_evaluation = csv.writer(combinefinal_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
combine_evaluation.writerow(['Evaluation for Combined Summary'])
combine_evaluation.writerow(['File Name', 'ROUGE', 'Method', 'Precision', 'Recall', 'F-measure'])

for f in os.listdir('sporttext'):
    generate_summary_kmeans('sporttext/'+f, 'samples/samplekmeans.txt')
    kmeans_scores = evaluate('samples/samplekmeans.txt', 'sportsummary/' + f)

    generate_summary_feature('sporttext/' + f, 'samples/sampleFeatures.txt')
    features_scores = evaluate('samples/sampleFeatures.txt', 'sportsummary/' + f)

    generate_summary_textrank('sporttext/'+f, 'samples/samplecosine.txt')
    textrank_scores = evaluate('samples/samplecosine.txt', 'sportsummary/' + f)

    sent_tokens_cosine = readFile_token('samples/samplecosine.txt')
    sent_tokens_kmeans = readFile_token('samples/samplekmeans.txt')
    sent_tokens_features = readFile_token('samples/sampleFeatures.txt')

    z = set(sent_tokens_cosine).union(sent_tokens_features)
    final = z.union(sent_tokens_kmeans)
    finalsummary = " ".join(final)
    outF = open('samples/finaloutput.txt', "w")
    outF.write(finalsummary)
    outF.close()
    final_scores = evaluate('samples/finaloutput.txt', 'sportsummary/' + f)
    for key in features_scores[0]:
        combine_evaluation.writerow([f, key, 'GraphBased', textrank_scores[0][key]['p'], textrank_scores[0][key]['r'], textrank_scores[0][key]['f']])
        combine_evaluation.writerow([f, key, 'FeatureBased', features_scores[0][key]['p'], features_scores[0][key]['r'], features_scores[0][key]['f']])
        combine_evaluation.writerow([f, key, 'ClusterBased', kmeans_scores[0][key]['p'], kmeans_scores[0][key]['r'], kmeans_scores[0][key]['f']])
        combine_evaluation.writerow([f, key, 'Final', final_scores[0][key]['p'], final_scores[0][key]['r'], final_scores[0][key]['f']])
