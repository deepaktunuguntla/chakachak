---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import os
import re

import gensim
import nltk; nltk.download(info_or_id=['stopwords'])
import pyLDAvis
import pyLDAvis.gensim_models

import pandas as pd

from pathlib import Path
from nltk.corpus import stopwords
```

### Download data

```python
if not Path('./data/nips_papers.csv').exists():
    Path('./data').mkdir()
    
!wget -O './data/nips_papers.csv' 'https://raw.githubusercontent.com/martenlienen/icml-nips-iclr-dataset/master/papers.csv'
```

### Load data

```python
papers = pd.read_csv('./data/nips_papers.csv')
```

```python
papers.columns
```

```python
papers.isna().any()
```

### Wrangle the data


#### Restructure

```python
papers['Title'] = papers['Title'].map(lambda x: x.lower())

papers = (papers
          .groupby(['Year', 'Title'])
          .agg({'Author': lambda x: x.tolist()})
          .reset_index()
         )
```

```python
papers.head()
```

#### Clean

```python
papers.Title = papers.Title.map(lambda x: re.sub('[:]','',x))
```

#### Remove stopwords

```python
def remove_stopwords(tokens: list, stop_words: list) -> list:
    return [ token for token in tokens if token not in stop_words ]

# Standard stop words
stop_words = stopwords.words('english')

# Tokenize the titles
tokenized_titles = [ nltk.word_tokenize(title) for title in papers.Title.values.tolist() ]

# Toeknized titles without stopwords
tokenized_titles_wo_stop_words = [ remove_stopwords(tokens, stop_words) for tokens in tokenized_titles ]
```

### Generate dictionary

```python
dictionary = gensim.corpora.Dictionary(tokenized_titles_wo_stop_words)
```

```python
'''
Convert tokens in titles to BOW format that is (token_id, token_count). 
Note that the gensim.corpora's Dictionary object generates ids for all the 
tokens, which we pass as an argument.

Example: 

['I', 'am', 'feeling', 'good'] -> [(0, 1), (1, 1), (2, 1), (3, 1)] 

where the tokens have the ids 

{'I': 0, 'am': 1, 'feeling': 2, 'good': 3}
'''

corpus = [ dictionary.doc2bow(title_tokens) for title_tokens in tokenized_titles_wo_stop_words ]
```

### LDA model

```python
# num of topics
num_topics = 10

# model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=num_topics)

# Print the keyword in the 10 topics
lda_model.print_topics()
```

```python
transformed_corpus = lda_model[corpus]
```

### Visualise the topics

```python
pyLDAvis_obj = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
```

```python
pyLDAvis.display(pyLDAvis_obj)
```

<!-- #region tags=[] -->
### References

- [Topic modelling in Python: Latent Dirichlet Allocation (LDA)](https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0)
<!-- #endregion -->
