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

import pandas as pd

from pathlib import Path
from wordcloud import WordCloud
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
papers
```

### Restructure data

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

### Wordcloud

```python
# Join all the titles
all_titles = ','.join(papers.Title.values)
```

```python
# create a wordcloud object
wordcloud = WordCloud(
    background_color='white',
    max_words=5000,
    contour_width=3,
    contour_color='steelblue'
)

# generate the wordcloud
wordcloud.generate(all_titles)

# visualise
wordcloud.to_image()
```

```python

```
