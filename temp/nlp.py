from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)
print(vectorizer)

corpus = [
     'This is the first document.',
     'This is the second second document.',
     'And the third one.',
     'Is this the first document?',
 ]
X = vectorizer.fit_transform(corpus)

analyze = vectorizer.build_analyzer()
# print(analyze("This is a text document to analyze."))

print(vectorizer.get_feature_names())

print(X.toarray())

# print(vectorizer.transform(['Something completely new.']).toarray())

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r"\b\w+\b", min_df=1)
analyze = bigram_vectorizer.build_analyzer()
print(analyze("Bi-grams are cool!"))

feature_index = bigram_vectorizer.vocabulary_.get('is this')
X_2[:, feature_index]

# array([0, 0, 0, 1]...)

X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
print(X_2)