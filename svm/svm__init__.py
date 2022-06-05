from builtins import print

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Set Random seed
# np.random.seed(500)

# Add the Data using pandas
Corpus = pd.read_csv(r"..\generated\data\lowFeature.csv", encoding='latin-1')
# print(Corpus)

# Step - 1a : Remove blank rows if any.
Corpus['lowFeatureList'].dropna(inplace=True)

# print(Corpus)
# Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
# Corpus['v2'] = [entry.lower() for entry in Corpus['v2']]
#
# # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
# Corpus['v2']= [word_tokenize(entry) for entry in Corpus['v2']]

# Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
# tag_map = defaultdict(lambda : wn.NOUN)
# tag_map['J'] = wn.ADJ
# tag_map['V'] = wn.VERB
# tag_map['R'] = wn.ADV
# for index,entry in enumerate(Corpus['v2']):
#     # Declaring Empty List to store the words that follow the rules for this step
#     Final_words = []
#     # Initializing WordNetLemmatizer()
#     word_Lemmatized = WordNetLemmatizer()
#     # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
#     for word, tag in pos_tag(entry):
#         # Below condition is to check for Stop words and consider only alphabets
#         if word not in stopwords.words('english') and word.isalpha():
#             word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
#             Final_words.append(word)
#     # The final processed set of words for each iteration will be stored in 'text_final'
#     Corpus.loc[index,'text_final'] = str(Final_words)


# Step - 2: Split the model into Train and Test Data set
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['lowFeatureList'],Corpus['Spam_Ham'],test_size=0.4, random_state=0)
#
# Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
#
#
#
#
# # Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
Tfidf_vect = TfidfVectorizer(max_features=1000)
Tfidf_vect.fit(Corpus['lowFeatureList'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# print('HERRRR')
# print(Train_X_Tfidf)
# print(' kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk ')
# print(Train_Y)
# Step - 5: Now we can run different algorithms to classify out data check for accuracy

from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# mlp.fit(Train_X_Tfidf,Train_Y)
#
# predictions = mlp.predict(Test_X_Tfidf)

from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(Test_Y,predictions))
# print(classification_report(Test_Y,predictions))


# Classifier - Algorithm - Naive Bayes
# fit the training dataset on the classifier
# Naive = naive_bayes.MultinomialNB()
# Naive.fit(Train_X_Tfidf,Train_Y)
#
# # predict the labels on validation dataset
# predictions_NB = Naive.predict(Test_X_Tfidf)
#
# # Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
# print('   ',classification_report(Test_Y, predictions_NB))
#
# print(confusion_matrix(Test_Y,predictions_NB))
# # print(classification_report(Test_Y, y_pred))
#
#
# # # Classifier - Algorithm - SVM
# # # fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=2, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)



# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print(' SVM  ',classification_report(Test_Y, predictions_SVM))

lr = LogisticRegression()
lr.fit(Train_X_Tfidf, Train_Y)

predictions_LR = lr.predict(Test_X_Tfidf)
# Cross validation
# accuracy = cross_validate(Random_Forest_model,Train_X_Tfidf,Train_Y,cv=10)['test_score']

print('Random accuracy is: ',accuracy_score(predictions_LR, Test_Y) * 100,'%')
print(' SVM  ',classification_report(Test_Y, predictions_LR))

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import cross_validate
# Random_Forest_model = RandomForestClassifier(n_estimators=100,criterion="entropy")
# Random_Forest_model.fit(Train_X_Tfidf, Train_Y, sample_weight=None)
#
# predictions_RF = Random_Forest_model.predict(Test_X_Tfidf)
# #Cross validation
# # accuracy = cross_validate(Random_Forest_model,Train_X_Tfidf,Train_Y,cv=10)['test_score']
# #
# # print('Random accuracy is: ',sum(accuracy)/len(accuracy)*100,'%')
# print("RF Accuracy Score -> ",accuracy_score(predictions_RF, Test_Y)*100)
#
#
#
# from sklearn import tree
# # #
# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf = clf.fit(Train_X_Tfidf,Train_Y)
#
# # SVM.fit(Train_X_Tfidf,Train_Y)
#
# # predict the labels on validation dataset
# predictions_DT = clf.predict(Test_X_Tfidf)
#
#
#
# # Use accuracy_score function to get the accuracy
# print("DT Accuracy Score -> ",accuracy_score(predictions_DT, Test_Y)*100)
# from sklearn.metrics import classification_report,confusion_matrix
# print(classification_report(Test_Y,predictions_DT))
# print(confusion_matrix(Test_Y,predictions_DT))
#
#
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#
# clf.fit(Train_X_Tfidf,Train_Y)
# MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
#               solver='lbfgs')
#
# predictions_FFNN = clf.predict(Test_X_Tfidf)
# print("FFNN Accuracy Score -> ",accuracy_score(predictions_FFNN, Test_Y)*100)
# # print(Test_X_Tfidf)
# # from sklearn.kernel_approximation import RBFSampler
# # from sklearn.linear_model import SGDClassifier
# #
# # rbf_feature = RBFSampler(gamma=1, random_state=1)
# # X_features = rbf_feature.fit_transform(Test_X_Tfidf)
# # clf = SGDClassifier(max_iter=5)
# # clf.fit(Train_X_Tfidf, Test_Y)
# # SGDClassifier(max_iter=100)
# # # clf.score(X_features, Test_Y)
# #
# # predictions_RB = clf.predict(Test_X_Tfidf)
# # print("RB Accuracy Score -> ",accuracy_score(predictions_RB, Test_Y)*100)
#
#
# # from sklearn.preprocessing import StandardScaler
# # scaler = StandardScaler()
# # scaler.fit(Train_X_Tfidf)
# # #
# # X_train = scaler.transform(Train_X)
# # X_test = scaler.transform(Test_X)
#
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
#                                          metric='cosine', metric_params=None, n_jobs=1)
# classifier.fit(Train_X_Tfidf,Train_Y)
#
#
# y_pred = classifier.predict(Test_X_Tfidf)
# #
# #
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(Test_Y,y_pred))
# print(classification_report(Test_Y, y_pred))
#
# classifier.fit(Train_X_Tfidf,Train_Y)
# predicted = classifier.predict(Test_X_Tfidf)
# acc = accuracy_score(predicted, Test_Y)
# print('KNN with TFIDF accuracy = ' + str(acc * 100) + '%')
# #
# # # scores = cross_val_score(knn, X_train, y_train, cv=3)
# # # print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# # # print(scores)
#
#
# # from sklearn.kernel_approximation import RBFSampler
# from sklearn.linear_model import SGDClassifier
# X = [[0, 0], [1, 1], [1, 0], [0, 1]]
# y = [0, 0, 1, 1]
# # rbf_feature = RBFSampler(gamma=1, random_state=1)
# # X_features = rbf_feature.fit_transform(X)
# clf = SGDClassifier(max_iter=5)
# clf.fit(Train_X_Tfidf,Train_Y)
# predicted = clf.predict(Test_X_Tfidf)
# SGDClassifier(max_iter=5)
# acc = accuracy_score(predicted, Test_Y)
# print('RB with TFIDF accuracy = ' + str(acc * 100) + '%')
#
# print(confusion_matrix(Test_Y,predicted))
# print(classification_report(Test_Y, predicted))
#
# into = 0
# Corpus1 = list(Corpus)
# Corpus2 = pd.read_csv(r"data\spam.csv",encoding='latin-1')
# print(Corpus2['v2'][5])
# # print(Corpus1)
# i = 0
# # model_results = {'Message': [],
# #                          'Original_Spam_Ham': [],
# #                          'Predicted': [],
# #                          'Spam_Ham': []
# #                          }
# # for i in range(len(Test_X)):
# #     if Corpus2['v1'][Test_X.index[i]] == 'ham':
# #         model_results['Message'].append(Corpus2['v2'][Test_X.index[i]])
# #         model_results['Original_Spam_Ham'].append('0')
# #         model_results['Predicted'].append(predicted[i])
# #         model_results['Spam_Ham'].append(Corpus2['v1'][Test_X.index[i]])
# #
# #         # print(Corpus2['v2'][Test_X.index[i]],'  0   ',predicted[i],'     ',Corpus2['v1'][Test_X.index[i]])
# #     elif Corpus2['v1'][Test_X.index[i]] == 'spam':
# #         model_results['Message'].append(Corpus2['v2'][Test_X.index[i]])
# #         model_results['Original_Spam_Ham'].append('1')
# #         model_results['Predicted'].append(predicted[i])
# #         model_results['Spam_Ham'].append(Corpus2['v1'][Test_X.index[i]])
# #         # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
# #     # i = i + 1
# #
# #
# # pd.DataFrame(model_results).to_csv('sms_results.csv', index=False)