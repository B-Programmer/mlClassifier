import os
import tkinter
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from builtins import print

import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
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
from sklearn.metrics import accuracy_score




mainDisplay = tkinter.Tk()
mainDisplay.geometry("850x600")

mainDisplay.title('A novel feature extraction approach in SMS spam filtering for mobile communication: One-Dimensional ternary patterns')

main1 = PanedWindow()
main1.pack(fill=BOTH, expand=1)

main2 = PanedWindow(main1, orient=VERTICAL)
main2.configure(background="green")
main3 = PanedWindow()

text = Text(main2, height=1, width=35)
text.insert(INSERT, "Select File...")
text.place(x=10, y=10)

var = StringVar()
selectLabel = Label(main2, textvariable=var, relief=FLAT)
var.set("<-- Click on Browse File Button to select file.")
selectLabel.place(x=380, y=8)
# var = StringVar()
# selectLabel = Label(main2, textvariable=var, relief=FLAT)
# var.set("--> ρ")
# selectLabel.place(x=650, y=10)
#
# textP = Text(main2, height=1, width=5)
# textP.insert(INSERT, "8")
# textP.place(x=688, y=10)

# var = StringVar()
# selectLabel = Label(main2, textvariable=var, relief=FLAT)
# var.set("-->B")
# selectLabel.place(x=750, y=10)
#
# textB = Text(main2, height=1, width=5)
# textB.insert(INSERT, "6")
# textB.place(x=788, y=10)




textSMS = Text(main2, height=5, width=90)
textSMS.insert(INSERT, "")
textSMS.place(x=120, y=40)


var = StringVar()
selectLabel = Label(main2, textvariable=var, relief=FLAT)
var.set("Select % to Test   ")
selectLabel.place(x=10, y=40)
spinList =Spinbox(main2, values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), width=14)
spinList.place(x=10, y=65)
# spinList.pack()

main1.add(main2)
#Function to Select SMS file (txt) and display it content(s)

global originalSMS
def displayBrowser():
    global originalSMS
    # Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    text.delete('1.0', END)
    text.place(x=10, y=10)
    textSMS.delete('1.0', END)
    textSMS.place(x=120, y=40)
    filename = askopenfilename()
    originalSMS = filename
    import os, fnmatch
    # pattern = "*.txt"
    #
    # if fnmatch.fnmatch(filename, pattern):
    text.insert(INSERT, filename)
    text.place(x=10, y=10)
    #     main1.add(main2)
    #     raw = open(filename, "r")  # Get the name of file from here to generate txt file
    #     mess = raw.read()
    #     textSMS.insert(INSERT, mess)
    #     originalSMS = mess
    # else:
    #     text.insert(INSERT, "Select File...")
    #     text.place(x=10, y=10)
    #     main1.add(main2)
    #     messagebox.showwarning('Error Message', 'File must be a text file')

global Train_X, Test_X, Train_Y, Test_Y, Train_X_Tfidf, Test_X_Tfidf, Corpus2
def selectionMode():
    # print(spinList.get())
    global originalSMS, Train_X, Test_X, Train_Y, Test_Y, Train_X_Tfidf
    roll = text.get('1.0', END)
    Corpus = pd.read_csv(originalSMS, encoding='latin-1')
    Corpus2 = pd.read_csv(originalSMS, encoding='latin-1')

    # Step - 1a : Remove blank rows if any.
    Corpus['v2'].dropna(inplace=True)

    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    Corpus['v2'] = [entry.lower() for entry in Corpus['v2']]

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus['v2'] = [word_tokenize(entry) for entry in Corpus['v2']]

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index, entry in enumerate(Corpus['v2']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index, 'text_final'] = str(Final_words)

    # Step - 2: Split the model into Train and Test Data set  test_size=float(spinList.get()),
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'], Corpus['v1'],
                                                                         test_size=0.4, random_state=0)

    # Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    # Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(Corpus['text_final'])

    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    Train_X = Train_X
    Test_X = Test_X
    Train_Y = Train_Y
    Test_Y = Test_Y
    Train_X_Tfidf = Train_X_Tfidf
    Test_X_Tfidf = Test_X_Tfidf
    result()
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Corpus21
    Train_X1 = Train_X
    Test_X1 = Test_X
    Train_Y1 = Train_Y
    Test_Y1 = Test_Y
    Train_X_Tfidf1 = Train_X_Tfidf
    Test_X_Tfidf1 = Test_X_Tfidf
    Corpus21 = Corpus2
    # # print(Corpus)





from sklearn.metrics import classification_report, confusion_matrix
def result():
    global Train_X_Tfidf, Train_Y, Test_X
    print(Train_X_Tfidf)
    # print(Test_X)

    print(Test_X)
    textSMS.insert(INSERT, Test_X)


def Naive_Bayes():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Corpus21
    try:
        if not os.path.exists("generated/"+spinList.get()+"/Naive"):
            os.makedirs("generated/"+spinList.get()+"/Naive")

    except OSError:
        print('Error: Creating Folder1')


    Naive = naive_bayes.GaussianNB()
    # MultinomialNB()
    Naive.fit(Train_X_Tfidf1, Train_Y1)

    # predict the labels on validation dataset
    predictions_NB = Naive.predict(Test_X_Tfidf1)




    # print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)
    # print(confusion_matrix(Test_Y, predictions_NB))
    # print(classification_report(Test_Y, predictions_NB))

    into = 0
    # Corpus1 = list(Corpus)
    # Corpus2 = pd.read_csv(r"data\spam.csv", encoding='latin-1')
    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_X1)):
        if Corpus21['v1'][Test_X1.index[i]] == 'ham':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_NB[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])

            # print(Corpus2['v2'][Test_X.index[i]],'  0   ',predicted[i],'     ',Corpus2['v1'][Test_X.index[i]])
        elif Corpus21['v1'][Test_X.index[i]] == 'spam':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_NB[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/'+spinList.get()+'/Naive/sms_results.csv', index=False)
    # Use accuracy_score function to get the accuracy
    f2 = open('generated/'+spinList.get()+'/Naive/NaiveResult.txt', 'a+')
    f2.write("Naive Bayes Accuracy Score -> " + str(round((accuracy_score(predictions_NB, Test_Y1) * 100), 2)) + "%\n")
    f2.write("Naive Bayes Confusion Matrix " + str(confusion_matrix(Test_Y1, predictions_NB)) + "\n")
    f2.write("Naive Bayes Classification Report " + str(classification_report(Test_Y1, predictions_NB)))
    f2.close()

    text = Text(main2, height=2, width=80)
    aa = "Naive Bayes Classification Report " + str(classification_report(Test_Y1, predictions_NB))
    text.insert(INSERT, str(aa).replace("'", "").replace("{", "").replace("}", "").replace("(", "").replace(")", ""))
    text.place(x=200, y=180)

def SVM():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Corpus21

    try:
        if not os.path.exists("generated/"+spinList.get()+"/SVM"):
            os.makedirs("generated/"+spinList.get()+"/SVM")

    except OSError:
        print('Error: Creating Folder1')
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf1, Train_Y1)

    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf1)

    # Use accuracy_score function to get the accuracy
    # print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)


    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_X1)):
        if Corpus21['v1'][Test_X1.index[i]] == 'ham':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_SVM[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])

            # print(Corpus2['v2'][Test_X.index[i]],'  0   ',predicted[i],'     ',Corpus2['v1'][Test_X.index[i]])
        elif Corpus21['v1'][Test_X1.index[i]] == 'spam':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_SVM[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/'+spinList.get()+'/SVM/sms_results.csv', index=False)

    f2 = open('generated/'+spinList.get()+'/SVM/SVMResult.txt', 'a+')
    f2.write("SVM Accuracy Score -> " + str(round((accuracy_score(predictions_SVM, Test_Y1) * 100), 2)) + "%\n")
    f2.write("SVM Confusion Matrix " + str(confusion_matrix(Test_Y1, predictions_SVM)) + "\n")
    f2.write("SVM Classification Report " + str(classification_report(Test_Y1, predictions_SVM)))
    f2.close()

    text = Text(main2, height=2, width=80)
    aa = "SVM Classification Report " + str(classification_report(Test_Y1, predictions_SVM))
    text.insert(INSERT, str(aa).replace("'", "").replace("{", "").replace("}", "").replace("(", "").replace(")", ""))
    text.place(x=200, y=220)

def Random_Forest():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Corpus21
    try:
        if not os.path.exists("generated/"+spinList.get()+"/RandomForest"):
            os.makedirs("generated/"+spinList.get()+"/RandomForest")

    except OSError:
        print('Error: Creating Folder1')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_validate
    Random_Forest_model = RandomForestClassifier(n_estimators=100, criterion="entropy")
    Random_Forest_model.fit(Train_X_Tfidf1, Train_Y1, sample_weight=None)

    predictions_RF = Random_Forest_model.predict(Test_X_Tfidf1)
    # Cross validation
    # accuracy = cross_validate(Random_Forest_model,Train_X_Tfidf,Train_Y,cv=10)['test_score']
    #
    # print('Random accuracy is: ',sum(accuracy)/len(accuracy)*100,'%')


    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_X1)):
        if Corpus21['v1'][Test_X1.index[i]] == 'ham':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_RF[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])

            # print(Corpus2['v2'][Test_X.index[i]],'  0   ',predicted[i],'     ',Corpus2['v1'][Test_X.index[i]])
        elif Corpus21['v1'][Test_X1.index[i]] == 'spam':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_RF[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/'+spinList.get()+'/RandomForest/sms_results.csv', index=False)

    print("RF Accuracy Score -> ", accuracy_score(predictions_RF, Test_Y1) * 100)
    f2 = open('generated/'+spinList.get()+'/RandomForest/RandomForestResult.txt', 'a+')
    f2.write("Random Forest Accuracy Score -> " + str(round((accuracy_score(predictions_RF, Test_Y1) * 100), 2)) + "%\n")
    f2.write("Random Forest Confusion Matrix " + str(confusion_matrix(Test_Y1, predictions_RF)) + "\n")
    f2.write("Random Forest Classification Report " + str(classification_report(Test_Y1, predictions_RF)))
    f2.close()

    text = Text(main2, height=2, width=80)
    aa = "Random Forest Classification Report " + str(classification_report(Test_Y1, predictions_RF))
    text.insert(INSERT, str(aa).replace("'", "").replace("{", "").replace("}", "").replace("(", "").replace(")", ""))
    text.place(x=200, y=260)

def Logistic_Regression():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Corpus21
    try:
        if not os.path.exists("generated/"+spinList.get()+"/LogisticRegression"):
            os.makedirs("generated/"+spinList.get()+"/LogisticRegression")

    except OSError:
        print('Error: Creating Folder1')
    # from sklearn import tree
    # #
    lr = LogisticRegression()
    lr.fit(Train_X_Tfidf1, Train_Y1)

    predictions_LR = lr.predict(Test_X_Tfidf1)

    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    # clf = clf.fit(Train_X_Tfidf1, Train_Y1)
    #
    # # SVM.fit(Train_X_Tfidf,Train_Y)
    #
    # # predict the labels on validation dataset
    # predictions_DT = clf.predict(Test_X_Tfidf1)

    # Use accuracy_score function to get the accuracy
    # print("DT Accuracy Score -> ", accuracy_score(predictions_DT, Test_Y) * 100)

    from sklearn.metrics import classification_report, confusion_matrix
    # print(classification_report(Test_Y, predictions_DT))
    # print(confusion_matrix(Test_Y, predictions_DT))



    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_X1)):
        if Corpus21['v1'][Test_X1.index[i]] == 'ham':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_LR[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])

            # print(Corpus2['v2'][Test_X.index[i]],'  0   ',predicted[i],'     ',Corpus2['v1'][Test_X.index[i]])
        elif Corpus21['v1'][Test_X1.index[i]] == 'spam':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_LR[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/'+spinList.get()+'/LogisticRegression/sms_results.csv', index=False)

    f2 = open('generated/'+spinList.get()+'/LogisticRegression/Logistic_Regression.txt', 'a+')
    f2.write("Logistic Regression Accuracy Score -> " + str(round((accuracy_score(predictions_LR, Test_Y1) * 100), 2)) + "%\n")
    f2.write("Logistic Regression Confusion Matrix " + str(confusion_matrix(Test_Y1, predictions_LR)) + "\n")
    f2.write("Logistic Regression Classification Report " + str(classification_report(Test_Y1, predictions_LR)))
    f2.close()

    text = Text(main2, height=2, width=80)
    aa = "Logistic Regression Classification Report " + str(classification_report(Test_Y1, predictions_LR))
    text.insert(INSERT, str(aa).replace("'", "").replace("{", "").replace("}", "").replace("(", "").replace(")", ""))
    text.place(x=200, y=300)

def Feed_ForwardNeural_Network():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Corpus21
    try:
        if not os.path.exists("generated/"+spinList.get()+"/FeedForwardNeuralNetwork"):
            os.makedirs("generated/"+spinList.get()+"/FeedForwardNeuralNetwork")

    except OSError:
        print('Error: Creating Folder1')

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(Train_X_Tfidf1, Train_Y1)
    MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
                  solver='lbfgs')

    predictions_FFNN = clf.predict(Test_X_Tfidf1)


    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_X1)):
        if Corpus21['v1'][Test_X1.index[i]] == 'ham':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_FFNN[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])

            # print(Corpus2['v2'][Test_X.index[i]],'  0   ',predicted[i],'     ',Corpus2['v1'][Test_X.index[i]])
        elif Corpus21['v1'][Test_X1.index[i]] == 'spam':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_FFNN[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/'+spinList.get()+'/FeedForwardNeuralNetwork/sms_results.csv', index=False)

    print("FFNN Accuracy Score -> ", accuracy_score(predictions_FFNN, Test_Y1) * 100)
    f2 = open('generated/'+spinList.get()+'/FeedForwardNeuralNetwork/FeedForwardNeuralNetworkResult.txt', 'a+')
    f2.write("FeedForward Neural Network Accuracy Score -> " + str(round((accuracy_score(predictions_FFNN, Test_Y1) * 100), 2)) + "%\n")
    f2.write("FeedForward Neural Network Confusion Matrix " + str(confusion_matrix(Test_Y1, predictions_FFNN)) + "\n")
    f2.write("FeedForward Neural Network Classification Report " + str(classification_report(Test_Y1, predictions_FFNN)))
    f2.close()
    text = Text(main2, height=2, width=80)
    aa = "FeedForward Neural Network Classification Report " + str(classification_report(Test_Y1, predictions_FFNN))
    text.insert(INSERT, str(aa).replace("'", "").replace("{", "").replace("}", "").replace("(", "").replace(")", ""))
    text.place(x=200, y=340)

def K_nearest_neighbors():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Corpus21
    try:
        if not os.path.exists("generated/"+spinList.get()+"/KNearestNeighbors"):
            os.makedirs("generated/"+spinList.get()+"/KNearestNeighbors")

    except OSError:
        print('Error: Creating Folder1')

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                      metric='cosine', metric_params=None, n_jobs=1)
    classifier.fit(Train_X_Tfidf1, Train_Y1)

    # y_pred = classifier.predict(Test_X_Tfidf)
    #
    #

    # print(confusion_matrix(Test_Y, y_pred))
    # print(classification_report(Test_Y, y_pred))

    classifier.fit(Train_X_Tfidf1, Train_Y1)
    predicted = classifier.predict(Test_X_Tfidf1)
    acc = accuracy_score(predicted, Test_Y1)


    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_X1)):
        if Corpus21['v1'][Test_X1.index[i]] == 'ham':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])

            # print(Corpus2['v2'][Test_X.index[i]],'  0   ',predicted[i],'     ',Corpus2['v1'][Test_X.index[i]])
        elif Corpus21['v1'][Test_X1.index[i]] == 'spam':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/'+spinList.get()+'/KNearestNeighbors/sms_results.csv', index=False)

    print('KNN with TFIDF accuracy = ' + str(acc * 100) + '%')

    f2 = open('generated/'+spinList.get()+'/KNearestNeighbors/KNearestNeigborResult.txt', 'a+')
    f2.write("K Nearest Neighbors Accuracy Score -> " + str(round((accuracy_score(predicted, Test_Y1) * 100), 2)) + "%\n")
    f2.write("K Nearest Neighbors Confusion Matrix " + str(confusion_matrix(Test_Y1, predicted)) + "\n")
    f2.write("K Nearest Neighbors Classification Report " + str(classification_report(Test_Y1, predicted)))
    f2.close()
    text = Text(main2, height=2, width=80)
    aa = "K Nearest Neighbors Classification Report " + str(classification_report(Test_Y1, predicted))
    text.insert(INSERT, str(aa).replace("'", "").replace("{", "").replace("}", "").replace("(", "").replace(")", ""))
    text.place(x=200, y=380)

def Radial_Basis():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Corpus21
    try:
        if not os.path.exists("generated/"+spinList.get()+"/RadialBasis"):
            os.makedirs("generated/"+spinList.get()+"/RadialBasis")

    except OSError:
        print('Error: Creating Folder1')
    from sklearn.linear_model import SGDClassifier

    # rbf_feature = RBFSampler(gamma=1, random_state=1)
    # X_features = rbf_feature.fit_transform(X)
    clf = SGDClassifier(max_iter=5)
    clf.fit(Train_X_Tfidf1, Train_Y1)
    predicted = clf.predict(Test_X_Tfidf1)
    SGDClassifier(max_iter=5)
    acc = accuracy_score(predicted, Test_Y1)


    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_X1)):
        if Corpus21['v1'][Test_X1.index[i]] == 'ham':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])

            # print(Corpus2['v2'][Test_X.index[i]],'  0   ',predicted[i],'     ',Corpus2['v1'][Test_X.index[i]])
        elif Corpus21['v1'][Test_X1.index[i]] == 'spam':
            model_results['Message'].append(Corpus21['v2'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpus21['v1'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/'+spinList.get()+'/RadialBasis/sms_results.csv', index=False)

    print('RB with TFIDF accuracy = ' + str(acc * 100) + '%')

    # print(confusion_matrix(Test_Y, predicted))
    # print(classification_report(Test_Y, predicted))
    f2 = open('generated/'+spinList.get()+'/RadialBasis/RadialBasisResult.txt', 'a+')
    f2.write("Radial Basis Accuracy Score -> " + str(round((accuracy_score(predicted, Test_Y1) * 100), 2)) + "%\n")
    f2.write("Radial Basis Confusion Matrix " + str(confusion_matrix(Test_Y1, predicted)) + "\n")
    f2.write("Radial Basis Classification Report " + str(classification_report(Test_Y1, predicted)))
    f2.close()

    text = Text(main2, height=2, width=80)
    aa = "Radial Basis Classification Report " + str(classification_report(Test_Y1, predicted))
    text.insert(INSERT, str(aa).replace("'", "").replace("{", "").replace("}", "").replace("(", "").replace(")", ""))
    text.place(x=200, y=420)









#Button to to select SMS file
browseButton = Button(mainDisplay, text="Browse File", height=1, command=displayBrowser)
browseButton.place(x=300, y=10)
#Button to extract Unwanted characters from SMS file


var = StringVar()
extractLabel = Label(main2, textvariable=var, relief=FLAT)
var.set("<-- Click to Split the model into Train and Test Data set")
extractLabel.place(x=298, y=130)

extractButton = Button(mainDisplay, text="Split the model into Train and Test Data set", height=1, command=selectionMode)
extractButton.place(x=10, y=130)


extractButton = Button(mainDisplay, text="Naive Bayes", height=1, command=Naive_Bayes)
extractButton.place(x=10, y=180)

extractButton = Button(mainDisplay, text="SVM", height=1, command=SVM)
extractButton.place(x=10, y=220)

extractButton = Button(mainDisplay, text="Random Forest", height=1, command=Random_Forest)
extractButton.place(x=10, y=260)

extractButton = Button(mainDisplay, text="Decision Tree", height=1, command=Logistic_Regression)
extractButton.place(x=10, y=300)

extractButton = Button(mainDisplay, text="Feed ForwardNeural Network", height=1, command=Feed_ForwardNeural_Network)
extractButton.place(x=10, y=340)

extractButton = Button(mainDisplay, text="K-nearest neighbors", height=1, command=K_nearest_neighbors)
extractButton.place(x=10, y=380)

extractButton = Button(mainDisplay, text="Radial Basis", height=1, command=Radial_Basis)
extractButton.place(x=10, y=420)

def drawChart():
    NB = open(r'generated/'+spinList.get()+'/Naive/NaiveResult.txt').readlines()
    SVM = open(r'generated/'+spinList.get()+'/SVM/SVMResult.txt').readlines()
    RForest = open(r'generated/'+spinList.get()+'/RandomForest/RandomForestResult.txt').readlines()
    DT = open(r'generated/'+spinList.get()+'/DecisionTree/DecisionTreeResult.txt').readlines()
    FeedFNN = open(r'generated/'+spinList.get()+'/FeedForwardNeuralNetwork/FeedForwardNeuralNetworkResult.txt').readlines()
    KNearN = open(r'generated/'+spinList.get()+'/KNearestNeighbors/KNearestNeigborResult.txt').readlines()
    RB = open(r'generated/'+spinList.get()+'/RadialBasis/RadialBasisResult.txt').readlines()

    import matplotlib.pyplot as pltt
    pltt.style.use('ggplot')
    print(int(float(NB[0].split('->')[1].replace('%', '').replace('\n', '').replace(' ', ''))))
    x = [ 'NB', 'SVM', 'RForest', 'DT', 'FeedFNN', 'KNear N', 'RB']
    mlResult = [int(float(NB[0].split('->')[1].replace('%', '').replace('\n', '').replace(' ', ''))), int(float(SVM[0].split('->')[1].replace('%', '').replace('\n', '').replace(' ', ''))),
              int(float(RForest[0].split('->')[1].replace('%', '').replace('\n', '').replace(' ', ''))),int(float(DT[0].split('->')[1].replace('%', '').replace('\n', '').replace(' ', ''))),
              int(float(FeedFNN[0].split('->')[1].replace('%', '').replace('\n', '').replace(' ', ''))), int(float(KNearN[0].split('->')[1].replace('%', '').replace('\n', '').replace(' ', ''))), int(float(RB[0].split('->')[1].replace('%', '').replace('\n', '').replace(' ', '')))]

    x_pos = [i for i, _ in enumerate(x)]
    pltt.bar(x_pos, mlResult, color='green')
    pltt.xlabel("Machine Learning Algorithms")
    pltt.ylabel("Output ")
    pltt.title("Machine learning output from various algorithms")
    pltt.xticks(x_pos, x)
    pltt.savefig('generated/'+spinList.get()+'.png')
    pltt.show()
extractButton = Button(mainDisplay, text="Draw Chart", height=1, command=drawChart)
extractButton.place(x=10, y=460)







main1.configure(background='black')
mainDisplay.mainloop()
