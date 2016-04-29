from gensim.models import Word2Vec
from nltk.corpus import reuters

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = Word2Vec(reuters.sents(), workers=4, size=300)
model.save('Corpus/word.embedding')
