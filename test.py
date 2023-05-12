from urllib.request import urlopen
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

url = "https://spring-store-api.herokuapp.com/api/products/pageable?page=0&size=4"
response = urlopen(url)
data_json = json.loads(response.read())
# print(data_json)
data_file = data_json['content']
df = pd.json_normalize(data_file, max_level=1)
# print(df)
df.to_csv('output.csv', index=False, encoding='utf-8')

shoes = pd.read_csv('output.csv')
new = shoes.drop(columns=['createdBy','createdDate','modifiedBy','modifiedDate','status','images','productInfors'])
cv = CountVectorizer(max_features=5000,stop_words='english')
# vector.shape
vector = cv.fit_transform(new['description']).toarray()
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
print(recommend("Adidas NMD"))