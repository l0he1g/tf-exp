# -*- coding: utf-8 -*- 

from sklearn.feature_extraction.text import CountVectorizer

docs = ["习近平致信祝贺金砖国家", "习近平会见德国社会民主党主席舒尔茨", "习近平会见韩国总统文在寅"]

cv = CountVectorizer()
ds = cv.fit_transform(docs)
print(ds)


