Summarization Project
===

Goal is to implement then improve MMR algorithm.
MMR - Maximal Marginal Relevance

Note
---
The diversity parameter doesn't affect the ROUGE score. I'm not sure if
it is implemented properly.

Current ROUGE scores
| F_Score | Precision | Recall |
|---:|---:|---:|---:|
|ROUGE 1   | 0.349120 | 0.355460 | 0.345130
|ROUGE 2   | 0.076890 | 0.078720 | 0.075770
|ROUGE su4 | 0.118160 | 0.120600 | 0.116640



Todo
---
Improve MMR

Installation
---
Install python 2.7
- Numpy
- NLTK

Can install both with either python-pip or anaconda
    pip install numpy nltk
    conda install numpy nltk

Open a python interpreter and type in

```python2
import nltk
nltk.download()
```

Download all of the book collection

Should have a working install.
