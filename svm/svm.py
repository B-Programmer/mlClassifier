# from builtins import print
#
# import pandas as pd
# import numpy as np
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.preprocessing import LabelEncoder
# from collections import defaultdict
# from nltk.corpus import wordnet as wn
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import model_selection, naive_bayes, svm
# from sklearn.metrics import accuracy_score
#
# #Set Random seed
# np.random.seed(500)
#
# # Add the Data using pandas
# Corpus = pd.read_csv(r"data\corpus.csv",encoding='latin-1')
# # print(Corpus)
# # Step - 1: Data Pre-processing - This will help in getting better results through the classification algorithms
#
# # Step - 1a : Remove blank rows if any.
# Corpus['text'].dropna(inplace=True)
#
# # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
# Corpus['text'] = [entry.lower() for entry in Corpus['text']]
#
# # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
# Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
#
# # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
#
# # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
# tag_map = defaultdict(lambda : wn.NOUN)
# tag_map['J'] = wn.ADJ
# tag_map['V'] = wn.VERB
# tag_map['R'] = wn.ADV
#
#
# for index,entry in enumerate(Corpus['text']):
#     # Declaring Empty List to store the words that follow the rules for this step
#     Final_words = []
#     # Initializing WordNetLemmatizer()
#     word_Lemmatized = WordNetLemmatizer()
#     # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
#     for word, tag in pos_tag(entry):
#         # Below condition is to check for Stop words and consider only alphabets
#         if word not in stopwords.words('english') and word.isalpha():
#             word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
#             Final_words.append(word_Final)
#     # The final processed set of words for each iteration will be stored in 'text_final'
#     Corpus.loc[index,'text_final'] = str(Final_words)
#
# #print(Corpus['text_final'].head())
#
# # Step - 2: Split the model into Train and Test Data set
# Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.3)
#
# # Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
# Encoder = LabelEncoder()
# Train_Y = Encoder.fit_transform(Train_Y)
# Test_Y = Encoder.fit_transform(Test_Y)
#
# # Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
# Tfidf_vect = TfidfVectorizer(max_features=5000)
# Tfidf_vect.fit(Corpus['text_final'])
#
# Train_X_Tfidf = Tfidf_vect.transform(Train_X)
# Test_X_Tfidf = Tfidf_vect.transform(Test_X)
#
#
# # Step - 5: Now we can run different algorithms to classify out data check for accuracy
#
# # Classifier - Algorithm - Naive Bayes
# # fit the training dataset on the classifier
# Naive = naive_bayes.MultinomialNB()
# Naive.fit(Train_X_Tfidf,Train_Y)
#
# # predict the labels on validation dataset
# predictions_NB = Naive.predict(Test_X_Tfidf)
#
# # Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
#
#
# # Classifier - Algorithm - SVM
# # fit the training dataset on the classifier
# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# SVM.fit(Train_X_Tfidf,Train_Y)
#
# # predict the labels on validation dataset
# predictions_SVM = SVM.predict(Test_X_Tfidf)
#
#
# # Use accuracy_score function to get the accuracy
# print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

import pandas as pd
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

# file = pd.read_csv(r"data\spam.csv",encoding='latin-1')

training_data = pd.read_csv(r"data\spam.csv",encoding='latin-1')    #util.get_parser_data("amazon.csv")
X_train, X_test, y_train, y_test = model_selection.train_test_split(training_data["sentence"], training_data["sentiment"],
                                                        test_size=0.2, random_state=5)
X_train, X_test = vectorize.createTFIDF(X_train, X_test, remove_stopwords=True, lemmatize=True, stemmer=False)
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                         metric='cosine', metric_params=None, n_jobs=1)

knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print('KNN with TFIDF accuracy = ' + str(acc * 100) + '%')

scores = cross_val_score(knn, X_train, y_train, cv=3)
print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)