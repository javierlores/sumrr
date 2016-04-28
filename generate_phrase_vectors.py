#! /usr/bin/env python2.7

import collections
import itertools
import xml.etree.ElementTree
import nltk
import re
import os
import numpy as np
import csv
import subprocess

# Note xml parser crashes on html entities (&amp; &lt; &gt; &eq;)
# All entities are replaced with '' emtpy string
strip_html_entities = re.compile(r'&[A-Za-z]+;')

# basic list of stopwords
stopwords = nltk.corpus.stopwords.words('english')

# returns a list of sentences and their corresponding tokens
def parse(filepath):
    with open(filepath) as f:
            data = f.read()
    data = strip_html_entities.sub('', data)
    text = xml.etree.ElementTree.fromstring(data).find('TEXT').text

    tokens = []
    sentences = nltk.tokenize.sent_tokenize(text)

    sentences = [sentence.replace('\n', '') for sentence in sentences]
    for sentence in sentences:
        words = nltk.tokenize.word_tokenize(sentence)
        words = [word.lower() for word in words]
        words = [word for word in words if word not in stopwords and word.isalnum()]
        tokens.append(words)
    return sentences, tokens


def save_phrase_vectors(casepath):
    inc = itertools.count()
    vocab = collections.defaultdict(lambda: inc.next())
    data = []
    text = []
    vec = []
    for _, _, files in os.walk(casepath):
        # ignore annoying apple files
        mask = ['._', '.DS_Store']
        valid_files = (file for file in files if not any(s in file for s in mask))
        for file in valid_files:
            filepath = os.path.join(casepath, file)
            a, b = parse(filepath)
            text += a
            data += b

    # Convert text to numpy array for easy indexing
    text_lens = np.array(map(len, text))
    text = np.array(text)

    # Change the cwd to properly run RAE to create phrase vectors
    # matlab needs to be working in the RAE directory
    cwd = os.getcwd()
    os.chdir('codeRAEVectorsNIPS2011')

    # Run each sentence individually
    # Otherwise Stanford parse tree runs out of memory
    phrase_vectors = []
    for i in range(0, len(text)):
        # Create input file to create phrase vectors
        with open('input.txt', 'w') as file:
            file.write(text[i]+'\n')

        # Execute script
        subprocess.call(['./phrase2Vector.sh'])

        # Read phrase vectors
        with open('outVectors.txt', 'r') as file:
            for line in file:
                phrase_vector = np.array(line.split(","), dtype='float32')
                phrase_vectors.append(phrase_vector)

    os.chdir(cwd)

    if not os.path.exists('Phrase_Vectors'):
        os.makedirs('Phrase_Vectors')

    # Write the summary to a file
    filename = casepath.split('/')[1]
    # Write the phrase vectors to a file
    with open('Phrase_Vectors/'+filename+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for phrase_vector in phrase_vectors:
            writer.writerow(phrase_vector)


for root, cases, _ in os.walk('Documents'):
    for case in cases:
        casepath = os.path.join(root, case)
        save_phrase_vectors(casepath)


