import os
import csv

from nltk import sent_tokenize
from trycosinewithpagerank import generate_summary_textrank
from features import generate_summary_feature
from kmeans import generate_summary_kmeans

from rougeurdu import evaluate

filepath = '../BBC News Summary/News Articles/business'
summarypath = '../BBC News Summary/Summaries/business'

# textrank_file  = open('textrank_file.csv', mode='w')
# textrank_evaluation = csv.writer(textrank_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# textrank_evaluation.writerow(['Evaluation for Textrank Summary'])
# textrank_evaluation.writerow(['File Name', 'ROUGE', 'Precision', 'Recall', 'F-measure'])
#
# for f in os.listdir(filepath):
#     generate_summary_textrank(filepath+'/'+f, 'samplecosine.txt')
#     scores = evaluate('samplecosine.txt', summarypath+'/'+f)
#     scores[0] = dict(scores[0])
#     for key in scores[0]:
#         textrank_evaluation.writerow([f, key, scores[0][key]['p'], scores[0][key]['r'], scores[0][key]['f']])

features_file  = open('features_file.csv', mode='w')
features_evaluation = csv.writer(features_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
features_evaluation.writerow(['Evaluation for Feature based Summary'])
features_evaluation.writerow(['File Name', 'ROUGE', 'Precision', 'Recall', 'F-measure'])

for f in os.listdir(filepath):
    print(f)
    generate_summary_feature(filepath+'/'+f, 'sampleFeatures.txt')
    scores = evaluate('sampleFeatures.txt', summarypath+'/'+f)
    scores[0] = dict(scores[0])
    for key in scores[0]:
        features_evaluation.writerow([f, key, scores[0][key]['p'], scores[0][key]['r'], scores[0][key]['f']])

# kmeans_file  = open('kmeans_file.csv', mode='w')
# kmeans_evaluation = csv.writer(kmeans_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# kmeans_evaluation.writerow(['Evaluation for kmeans Summary'])
# kmeans_evaluation.writerow(['File Name', 'ROUGE', 'Precision', 'Recall', 'F-measure'])
#
# for f in os.listdir(filepath):
#     generate_summary_kmeans(filepath+'/'+f, 'samplekmeans.txt')
#     scores = evaluate('samplekmeans.txt', summarypath+'/'+f)
#     scores[0] = dict(scores[0])
#     for key in scores[0]:
#         kmeans_evaluation.writerow([f, key, scores[0][key]['p'], scores[0][key]['r'], scores[0][key]['f']])

