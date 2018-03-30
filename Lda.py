from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

corpus = []
for line in open('net_news_jieba.txt', 'r').readlines():
    corpus.append(line.strip())

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus)
analyze = vectorizer.build_analyzer()
tf = x.toarray()

lda = LatentDirichletAllocation(n_components=5, max_iter=5)
lda.fit(tf)

feature_name = vectorizer.get_feature_names()

for topicidx, topic in enumerate(lda.components_):
    msg="Topic #%d: "%topicidx
    msg+=" ".join([feature_name[i] for i in topic.argsort()[:-20-1:-1]])
    print(msg)