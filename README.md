Summarization Project
===

Goal is to implement then improve MMR algorithm.
MMR - Maximal Marginal Relevance

Note
---
The diversity parameter doesn't affect the ROUGE score. I'm not sure if
it is implemented properly.

Current ROUGE scores

| | Precision | Precision | F Score |
|---|---:|---:|---:|
| ROUGE 1   | 0.360200 | 0.363660 | 0.361890 |
| ROUGE 2   | 0.086020 | 0.086990 | 0.086490 |
| ROUGE su4 | 0.125500 | 0.126780 | 0.126120 |

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
