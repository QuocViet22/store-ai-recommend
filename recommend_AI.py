import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

shoes = pd.read_csv('./input/test.csv')
# shoes.head(2)
# shoes.shape
# shoes.head()

cv = CountVectorizer(max_features=5000,stop_words='english')

new = shoes
vector = cv.fit_transform(new['description']).toarray()

# vector.shape

similarity = cosine_similarity(vector)

# similarity

def recommend(shoe):
    index = new[new['name'] == shoe].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    s = ""
    for i in distances[1:6]:
        s = s + str(new.iloc[i[0]].id) + ","
        print (new.iloc[i[0]].id)
    return s

# print(recommend("Adidas NMD"))