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

<!-- #region tags=[] -->
### Load dependencies
<!-- #endregion -->

```python
import bs4 as bs
import urllib.request
import re
import nltk; nltk.download(info_or_id=['stopwords','punkt'])

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
```

<!-- #region tags=[] -->
### Scrape a wikipedia article
<!-- #endregion -->

```python
scrapped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')
article = scrapped_data.read()
```

<!-- #region tags=[] -->
### Parse using lxml and get paragraphs
<!-- #endregion -->

```python
parsed_article = bs.BeautifulSoup(article,'lxml')
```

```python
paragraphs = parsed_article.find_all('p')
article_text = ""
for p in paragraphs:
    article_text += p.text
```

<!-- #region tags=[] -->
### Process the article text
<!-- #endregion -->

```python
processed_article = article_text.lower()
```

```python
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)
```

```python
processed_article = re.sub(r'\s+', ' ', processed_article)
```

<!-- #region tags=[] -->
### Prepare the dataset
<!-- #endregion -->

```python
all_sentences = nltk.sent_tokenize(processed_article)
```

```python
all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
```

```python
# Removing stop words
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]
```

<!-- #region tags=[] -->
### Creating Word2Vec model
<!-- #endregion -->

```python
word2vec = Word2Vec(all_words, min_count=2)
```

```python
vocabulary = word2vec.wv.index_to_key
print(f'vocabulary: {vocabulary}')
print(f'vocabulary size: {len(vocabulary)}')
```

<!-- #region tags=[] -->
### Model analysis
<!-- #endregion -->

```python
# Get the embedding for 'artificial'
word2vec.wv['artificial']
```

```python
# Get most similar words
word2vec.wv.most_similar('intelligence')
```

<!-- #region tags=[] -->
### Visualise embeddings
<!-- #endregion -->

```python
X = word2vec.wv.vectors
```

```python
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
```

```python
pyplot.scatter(pca_result[:,0], pca_result[:,1])
for i, word in enumerate(vocabulary):
    pyplot.annotate(word, xy=(pca_result[i, 0], pca_result[i, 1]))
pyplot.show()
```

```python

```
