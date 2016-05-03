# Tyler Truong 3221610
# two
# three
# Professor Fei Liu
# NLP
import collections
import itertools
import xml.etree.ElementTree
import nltk
import numpy as np
import os
import re
from gensim.models import Word2Vec

# load Word2Vec
w2v = Word2Vec.load('Corpus/word.embedding')

DIM = 300

# extracted from average char length of human summaries
SUMMARY_LENGTH = 650

# doesn't appear to do anything. The length might be too small, so there
# is not enough room for deviations to happen.
DIVERSITY = 0.7

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

    features = []
    sentences = nltk.tokenize.sent_tokenize(text)

    sentences = [sentence.replace('\n', '') for sentence in sentences]
    for sentence in sentences:
        words = nltk.tokenize.word_tokenize(sentence)
        words = [word.lower() for word in words]
        words = [word for word in words if word not in stopwords and word.isalnum()]
        words = [w2v[word] for word in words if word in w2v]
        if not words:
            words = [np.zeros(DIM), np.zeros(DIM)]
        words = reduce(lambda x, y: x + y, words)
        features.append(words)
    return sentences, features

# performs summarization on each case
def summarize(casepath, sum_len, Y):
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

    # build matrix of (sentence, feature)
    mat = np.vstack(data)

    # term weighting - IDF
    tf = mat
    idf = np.log(float(mat.shape[1]) / (mat.sum(axis=0) + 1)**0.5)
    mat = tf * idf

    # MMR
    # centroid is average of all sentences, used as Query parameter
    # sim1 an array of all cosine sims between sentences and centroid
    # sim2 square matrix of each row's similarity to each other
    #
    # idx - index vector
    # selected - index mask
    #   for masking sim1 and sim2 thoughout MMR iteration.
    #   selected keeps track of what sentences are in the summary
    #   each iteration, mmr computes max(out) - max(out, in)
    centroid = mat.mean(axis=0)
    sim1 = sim_mat_vec(mat, centroid)
    sim2 = sim_mat(mat)
    idx = np.arange(len(text))
    selected = np.zeros(len(text), dtype=bool)

    # initial iteration
    pick = sim1.argmax()
    selected[pick] = True
    summr_len = text_lens[pick]

    # stop when summary is long enough or all sentences selected
    while summr_len < sum_len and not np.all(selected):
        a = sim1[idx[~selected]]
        b = sim2[idx[~selected]]
        b = b[:,idx[selected]].max()
        # following line is very subtle
        # argmax computes index out of selected items e.g. (3, 5, 2) index 1
        # use argmax index to select actual index in index array
        # (3, 5, 2)[1] == 5
        pick = idx[~selected][(Y * a - (1-Y) * b).argmax()]
        selected[pick] = True
        summr_len += text_lens[pick]
    summr = text[idx[selected]]

    # Create the folder to hold the system summaries if it doesn't exist
    if not os.path.exists('System_Summaries'):
       os.makedirs('System_Summaries')

    # Write the summary to a file
    filename = casepath.split('/')[1]
    with open('System_Summaries/'+filename, 'w') as file:
        file.write(' '.join(summr.tolist()))


# compute cosine sim between all rows of matrix A against B vector
def sim_mat_vec(A, B):
    top = (A * B).sum(axis=1)
    bottom = ((A**2).sum(axis=1)**0.5) * ((B**2).sum()**0.5)
    sim = top / (bottom + 1)
    return sim

# compute cosine sim between all rows of matrix A with each other
def sim_mat(A):
    top = np.dot(A, A.T)
    bottom = (np.diag(top)**0.5 + 1)
    sim = top / bottom
    sim = sim.T / bottom
    np.fill_diagonal(sim, 0)
    return sim


# main iteration through all of the directories.
for root, cases, _ in os.walk('Documents'):
    for case in cases:
        casepath = os.path.join(root, case)
        summarize(casepath, SUMMARY_LENGTH, DIVERSITY)
