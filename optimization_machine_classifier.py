import os
import tkinter
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
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
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import random
import math


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
###########################################################################################
###########################################################################################

selectLabel.place(x=380, y=8)
var = StringVar()
selectLabel = Label(main2, textvariable=var, relief=FLAT)
var.set("--> ρ")
selectLabel.place(x=650, y=10)

textP = Text(main2, height=1, width=5)
textP.insert(INSERT, "8")
textP.place(x=688, y=10)

var = StringVar()
selectLabel = Label(main2, textvariable=var, relief=FLAT)
var.set("-->β")
selectLabel.place(x=750, y=10)

textB = Text(main2, height=1, width=5)
textB.insert(INSERT, "6")
textB.place(x=788, y=10)

###########################################################################################
###########################################################################################
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



global originalSMSList

def displayBrowser():
    global originalSMSList
    # Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    text.delete('1.0', END)
    text.place(x=10, y=10)
    textSMS.delete('1.0', END)
    textSMS.place(x=120, y=40)
    filename = askopenfilename()
    originalSMSList = filename
    import os, fnmatch
    # pattern = "*.txt"
    #
    # if fnmatch.fnmatch(filename, pattern):
    text.insert(INSERT, filename)
    text.place(x=10, y=10)


upFeaturesList, lowFeaturesList = [], []
global originalSMS, originalSMS1
global Train_X, Test_X, Train_Y, Test_Y, Train_X_Tfidf, Test_X_Tfidf, Train_Xlow, Test_Xlow, Train_Ylow, Test_Ylow, Train_X_Tfidflow, Test_X_Tfidflow, Corpus2, Corpuslow2
def selectionMode():
    global Train_X, Test_X, Train_Y, Test_Y, Train_X_Tfidf, Test_X_Tfidf, Train_Xlow, Test_Xlow, Train_Ylow, Test_Ylow, Train_X_Tfidflow, Test_X_Tfidflow
    global originalSMSList
    try:
        if not os.path.exists("generated/data"):
            os.makedirs("generated/data")

    except OSError:
        print('Error: Creating Folder1')
    # print(originalSMSList)
    Corpus = pd.read_csv(originalSMSList, encoding='latin-1')
    # Corpus['v2'].drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    # Corpus['v2'].dropna(inplace=True)
    Corpus['v2'].dropna(inplace=True)
    # print(Corpus['v2'])
    x_start = [int(textP.get('1.0', END)), int(textB.get('1.0', END))]
    iLk = 0

    model_results_upFeature = {'Message': [],
                               'upFeatureList': [],
                               'Spam_Ham': []
                               }
    model_results_lowFeature = {'Message': [],
                               'lowFeatureList': [],
                               'Spam_Ham': []
                               }
    for lis in Corpus['v2']:

        originalSMS = lis.replace(" ", "")
        # originalSMS1 = originalSMS
        showSMS = "The given Message: ", originalSMS, " with length: ", len(originalSMS)
        # Set the initial value of P
        # print(showSMS)
        # textUnwanterSpace = Text(main2, height=5, width=103)
        # textUnwanterSpace.delete('1.0', END)
        # textUnwanterSpace.insert(INSERT, showSMS)
        # textUnwanterSpace.place(x=10, y=170)

        ##################################################
        # Simulated Annealing
        ##################################################
        # Number of cycles   50
        n = 2
        # Number of trials per cycle
        m = 2
        # Number of accepted solutions
        na = 0.0
        # Probability of accepting worse solution at the start
        p1 = 0.7
        # Probability of accepting worse solution at the end
        p50 = 0.001
        # Initial temperature
        t1 = -1.0 / math.log(p1)
        # Final temperature
        t50 = -1.0 / math.log(p50)
        # Fractional reduction every cycle
        frac = (t50 / t1) ** (1.0 / (n - 1.0))
        # Initialize x
        x = np.zeros((n + 1, 2))
        x[0] = x_start
        xi = np.zeros(2)
        xi = x_start
        na = na + 1.0
        # Current best results so far
        xc = np.zeros(2)
        xc = x[0]
        fc = f(xi, 0, n, 0, originalSMS)
        fs = np.zeros(n + 1)
        fs[0] = fc
        # Current temperature
        t = t1
        # DeltaE Average
        DeltaE_avg = 0.0

        # fwriter = open("generated/bestObjective/optimization/best_objective.txt", "a+")

        for i in range(n):

            # print('Cycle: ' + str(i) + ' with Temperature: ' + str(t))
            # fwriter.write('Cycle: ' + str(i) + ' with Temperature: ' + str(t) + '\n')
            for j in range(m):
                # Generate new trial points
                xi[0] = xc[0] + (random.random() - 0.5) * 8
                xi[1] = xc[1] + (random.random() - 0.5) * 20
                # Clip to upper and lower bounds
                xi[0] = max(min(xi[0], 8.0), -8.0)
                xi[1] = max(min(xi[1], 20.0), -20.0)
                DeltaE = abs(f(xi, i, n, 1, originalSMS) - fc)
                if (f(xi, i, n, 1, originalSMS) > fc):
                    # Initialize DeltaE_avg if a worse solution was found
                    #   on the first iteration
                    if (i == 0 and j == 0): DeltaE_avg = DeltaE
                    # objective function is worse
                    # generate probability of acceptance

                    Del_avg_T = (DeltaE_avg * t)
                    if (DeltaE_avg * t) <= 0:
                        Del_avg_T = 1
                    p = math.exp(-DeltaE / Del_avg_T)
                    # determine whether to accept worse point
                    if (random.random() < p):
                        # accept the worse solution
                        accept = True
                    else:
                        # don't accept the worse solution
                        accept = False
                else:
                    # objective function is lower, automatically accept
                    accept = True
                if (accept == True):
                    # update currently accepted solution
                    xc[0] = xi[0]
                    xc[1] = xi[1]
                    fc = f(xc, i, n, 2, originalSMS)
                    # increment number of accepted solutions
                    na = na + 1.0
                    # update DeltaE_avg
                    DeltaE_avg = (DeltaE_avg * (na - 1.0) + DeltaE) / na
            # Record the best x values at the end of every cycle
            x[i + 1][0] = xc[0]
            x[i + 1][1] = xc[1]
            fs[i + 1] = fc
            # Lower the temperature for next cycle
            t = frac * t

        # Normalize the value of P
        if (xc[1] < 6):
            xc[1] = abs(xc[1]) + 6
        # print solution
        # print('Best solution with optimal value β is: ' + str(int(textP.get('1.0', END))) + ' and ρ is: ' + str(
        #     int(textB.get('1.0', END))))
        # aa = 'Best solution with optimal value β is: ' + str(int(textP.get('1.0', END))) + ' and ρ is: ' + str(
        #     int(textB.get('1.0', END)))
        #
        # # fwriter.write(str(''.join(aa).encode('utf-8')) + '\n')
        # text = Text(main2, height=2, width=50)
        # text.insert(INSERT,
        #             str(aa).replace("'", "").replace("{", "").replace("}", "").replace("(", "").replace(")", ""))
        # text.place(x=423, y=290)
        # # print('Best objective(1D-TP): ' + str(int(fc)))
        # aa = 'Best objective(1D-TP): ' + str(int(fc))
        # print(upFeatureListAll)


        ij = 0
        for upL in upFeatureListAll:
            ij = ij + 1
            if ij == len(upFeatureListAll) and upL != []:
                model_results_upFeature['Message'].append(lis)
                model_results_upFeature['upFeatureList'].append(str(upL).replace('[', "").replace(']', "").replace(',', ""))
                # print('ssssssssss ',Corpus['v1'][iLk])
                if Corpus['v1'][iLk] == 'ham':
                    model_results_upFeature['Spam_Ham'].append(0)
                if Corpus['v1'][iLk] == 'spam':
                    model_results_upFeature['Spam_Ham'].append(1)

                # print('LEXXXX ', upL, ' SIZE ', len(upFeatureListAll))
                # print('BBLEXXXX ', upL, ' SIZE ', len(lowFeaturesListAll))

        ij2 = 0
        for upL in lowFeaturesListAll:
            ij2 = ij2 + 1
            if ij2 == len(lowFeaturesListAll) and upL != []:
                model_results_lowFeature['Message'].append(lis)
                model_results_lowFeature['lowFeatureList'].append(
                    str(upL).replace('[', "").replace(']', "").replace(',', ""))
                # print('ssssssssss ',Corpus['v1'][iLk])
                if Corpus['v1'][iLk] == 'ham':
                    model_results_lowFeature['Spam_Ham'].append(0)
                if Corpus['v1'][iLk] == 'spam':
                    model_results_lowFeature['Spam_Ham'].append(1)

                # print('LEXXXX ', upL, ' SIZE ', len(upFeatureListAll))
                # print('BBLEXXXX ', upL, ' SIZE ', len(lowFeaturesListAll))
        # pd.DataFrame(model_results_upFeature).to_csv('generated/sms_results112.csv', index=True)
        # model_results_lowFeature = {'Message': [],
        #                            'lowFeatureList': [],
        #                            'Spam_Ham': []
        #                            }
        # ik = 0
        # for upL in lowFeaturesListAll:
        #     ik = ik + 1
        #     if ik == len(lowFeaturesListAll) and upL != []:
        #         print('BBLEXXXX ', upL, ' SIZE ', len(lowFeaturesListAll))

        # if upFeatureListAll[-1] == []:
        #     print(upFeatureListAll[-2])
        #     print(lowFeaturesListAll[-2])
        # else:
        #     print(upFeatureListAll[-1])
        #     print(lowFeaturesListAll[-1])

        # fwriter.write(aa + '\n')
        # fwriter.close()


        # print(lis)
        iLk = iLk + 1
    # print(model_results_upFeature)
    pd.DataFrame(model_results_upFeature).to_csv('generated/data/upFeature.csv', index=False)
    pd.DataFrame(model_results_lowFeature).to_csv('generated/data/lowFeature.csv', index=False)

    Corpus = pd.read_csv(r"generated/data/upFeature.csv", encoding='latin-1')
    Corpus2 = pd.read_csv(r"generated/data/upFeature.csv", encoding='latin-1')
    # print(Corpus)

    # Step - 1a : Remove blank rows if any.
    Corpus['upFeatureList'].dropna(inplace=True)

    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['upFeatureList'], Corpus['Spam_Ham'],
                                                                        test_size=0.4, random_state=0)

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    # Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
    Tfidf_vect = TfidfVectorizer(max_features=1000)
    Tfidf_vect.fit(Corpus['upFeatureList'])

    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    Train_X = Train_X
    Test_X = Test_X
    Train_Y = Train_Y
    Test_Y = Test_Y
    Train_X_Tfidf = Train_X_Tfidf
    Test_X_Tfidf = Test_X_Tfidf
    # result()
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Corpus21
    Train_X1 = Train_X
    Test_X1 = Test_X
    Train_Y1 = Train_Y
    Test_Y1 = Test_Y
    Train_X_Tfidf1 = Train_X_Tfidf
    Test_X_Tfidf1 = Test_X_Tfidf
    Corpus21 = Corpus2




    Corpuslow = pd.read_csv(r"generated/data/lowFeature.csv", encoding='latin-1')
    Corpuslow2 = pd.read_csv(r"generated/data/lowFeature.csv", encoding='latin-1')
    # print(Corpus)

    # Step - 1a : Remove blank rows if any.
    Corpuslow['lowFeatureList'].dropna(inplace=True)

    Train_Xlow, Test_Xlow, Train_Ylow, Test_Ylow = model_selection.train_test_split(Corpuslow['lowFeatureList'], Corpuslow['Spam_Ham'],
                                                                        test_size=0.4, random_state=0)

    Encoder = LabelEncoder()
    Train_Ylow = Encoder.fit_transform(Train_Ylow)
    Test_Ylow = Encoder.fit_transform(Test_Ylow)

    # Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
    Tfidf_vectlow = TfidfVectorizer(max_features=1000)
    Tfidf_vectlow.fit(Corpuslow['lowFeatureList'])

    Train_X_Tfidflow = Tfidf_vectlow.transform(Train_Xlow)
    Test_X_Tfidflow = Tfidf_vectlow.transform(Test_Xlow)

    Train_Xlow = Train_Xlow
    Test_Xlow = Test_Xlow
    Train_Ylow = Train_Ylow
    Test_Ylow = Test_Ylow
    Train_X_Tfidflow = Train_X_Tfidflow
    Test_X_Tfidflow = Test_X_Tfidflow
    # result()
    global Train_Xlow1, Test_Xlow1, Train_Ylow1, Test_Ylow1, Train_X_Tfidflow1, Test_X_Tfidflow1, Corpuslow21
    Train_Xlow1 = Train_Xlow
    Test_Xlow1 = Test_Xlow
    Train_Ylow1 = Train_Ylow
    Test_Ylow1 = Test_Ylow
    Train_X_Tfidflow1 = Train_X_Tfidflow
    Test_X_Tfidflow1 = Test_X_Tfidflow
    Corpuslow21 = Corpuslow2
    ''






from sklearn.metrics import classification_report, confusion_matrix

def Naive_Bayes():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Train_Xlow1, Test_Xlow1, Train_Ylow1, Test_Ylow1, Train_X_Tfidflow1, Test_X_Tfidflow1, Corpus21, Corpuslow21
    try:
        if not os.path.exists("generated/" + spinList.get() +"/Naive"):
            os.makedirs("generated/" + spinList.get()   + "/Naive")
            # os.makedirs("generated/" + spinList.get() + "/" + textB.get('1.0', END) + "/Naive")

    except OSError:
        print('Error: Creating Folder1')

    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf1, Train_Y1)

    # predict the labels on validation dataset
    predictions_NB = Naive.predict(Test_X_Tfidf1)

    into = 0
    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_X1)):

        if Corpus21['Spam_Ham'][Test_X1.index[i]] == 0:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_NB[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpus21['Spam_Ham'][Test_X1.index[i]] == 1:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_NB[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() +'/Naive/upFeature.csv', index=False)
    # Use accuracy_score function to get the accuracy
    f2 = open('generated/' + spinList.get() +'/Naive/upFeature.txt', 'a+')
    f2.write("Naive Bayes Accuracy Score -> " + str(round((accuracy_score(predictions_NB, Test_Y1) * 100), 2)) + "%\n")
    f2.write("Naive Bayes Confusion Matrix " + str(confusion_matrix(Test_Y1, predictions_NB)) + "\n")
    f2.write("Naive Bayes Classification Report " + str(classification_report(Test_Y1, predictions_NB)))
    f2.close()

    # text = Text(main2, height=2, width=80)
    # aa = "Naive Bayes Classification Report " + str(classification_report(Test_Y1, predictions_NB))
    # text.insert(INSERT, str(aa).replace("'", "").replace("{", "").replace("}", "").replace("(", "").replace(")", ""))
    # text.place(x=200, y=180)

    # Naive = naive_bayes.MultinomialNB()
    # Naive.fit(Train_X_Tfidf1, Train_Y1)
    #
    # # predict the labels on validation dataset
    # predictions_NB = Naive.predict(Test_X_Tfidf1)
    #
    # print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)
    # print(confusion_matrix(Test_Y, predictions_NB))
    # print(classification_report(Test_Y, predictions_NB))

    # Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidflow1, Train_Ylow1)

    # predict the labels on validation dataset
    predictions_NB = Naive.predict(Test_X_Tfidflow1)

    into = 0
    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_Xlow1)):

        if Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 0:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_NB[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 1:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_NB[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get()+'/Naive/lowFeature.csv', index=False)
    # Use accuracy_score function to get the accuracy
    f2 = open('generated/' + spinList.get()+ '/Naive/lowFeature.txt', 'a+')
    f2.write("Naive Bayes Accuracy Score -> " + str(round((accuracy_score(predictions_NB, Test_Ylow1) * 100), 2)) + "%\n")
    f2.write("Naive Bayes Confusion Matrix " + str(confusion_matrix(Test_Ylow1, predictions_NB)) + "\n")
    f2.write("Naive Bayes Classification Report " + str(classification_report(Test_Ylow1, predictions_NB)))
    f2.close()
    ''

def SVM():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Train_Xlow1, Test_Xlow1, Train_Ylow1, Test_Ylow1, Train_X_Tfidflow1, Test_X_Tfidflow1, Corpus21, Corpuslow21

    try:
        if not os.path.exists("generated/" + spinList.get() + "/SVM"):
            os.makedirs("generated/" + spinList.get() + "/SVM")

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
        if Corpus21['Spam_Ham'][Test_X1.index[i]] == 0:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_SVM[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpus21['Spam_Ham'][Test_X1.index[i]] == 1:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_SVM[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() + '/SVM/upFeature.csv', index=False)

    f2 = open('generated/' + spinList.get() + '/SVM/upFeature.txt', 'a+')
    f2.write("SVM Accuracy Score -> " + str(round((accuracy_score(predictions_SVM, Test_Y1) * 100), 2)) + "%\n")
    f2.write("SVM Confusion Matrix " + str(confusion_matrix(Test_Y1, predictions_SVM)) + "\n")
    f2.write("SVM Classification Report " + str(classification_report(Test_Y1, predictions_SVM)))
    f2.close()

    ##################################################################################
    ##################################################################################
    SVM.fit(Train_X_Tfidflow1, Train_Ylow1)

    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidflow1)

    # Use accuracy_score function to get the accuracy
    # print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)

    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }

    for i in range(len(Test_Xlow1)):
        if Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 0:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_SVM[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 1:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_SVM[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() + '/SVM/lowFeature.csv', index=False)

    f2 = open('generated/' + spinList.get() + '/SVM/lowFeature.txt', 'a+')
    f2.write("SVM Accuracy Score -> " + str(round((accuracy_score(predictions_SVM, Test_Ylow1) * 100), 2)) + "%\n")
    f2.write("SVM Confusion Matrix " + str(confusion_matrix(Test_Ylow1, predictions_SVM)) + "\n")
    f2.write("SVM Classification Report " + str(classification_report(Test_Ylow1, predictions_SVM)))
    f2.close()



    ''

def Random_Forest():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Train_Xlow1, Test_Xlow1, Train_Ylow1, Test_Ylow1, Train_X_Tfidflow1, Test_X_Tfidflow1, Corpus21, Corpuslow21

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
        if Corpus21['Spam_Ham'][Test_X1.index[i]] == 0:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_RF[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpus21['Spam_Ham'][Test_X1.index[i]] == 1:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_RF[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/'+spinList.get()+'/RandomForest/upFeatureList.csv', index=False)

    print("RF Accuracy Score -> ", accuracy_score(predictions_RF, Test_Y1) * 100)
    f2 = open('generated/'+spinList.get()+'/RandomForest/upFeatureList.txt', 'a+')
    f2.write("Random Forest Accuracy Score -> " + str(round((accuracy_score(predictions_RF, Test_Y1) * 100), 2)) + "%\n")
    f2.write("Random Forest Confusion Matrix " + str(confusion_matrix(Test_Y1, predictions_RF)) + "\n")
    f2.write("Random Forest Classification Report " + str(classification_report(Test_Y1, predictions_RF)))
    f2.close()

    Random_Forest_model.fit(Train_X_Tfidflow1, Train_Ylow1, sample_weight=None)

    predictions_RF = Random_Forest_model.predict(Test_X_Tfidflow1)
    # Cross validation
    # accuracy = cross_validate(Random_Forest_model,Train_X_Tfidf,Train_Y,cv=10)['test_score']
    #
    # print('Random accuracy is: ',sum(accuracy)/len(accuracy)*100,'%')

    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_Xlow1)):
        if Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 0:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_RF[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 1:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_RF[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() + '/RandomForest/lowFeatureList.csv', index=False)

    print("RF Accuracy Score -> ", accuracy_score(predictions_RF, Test_Ylow1) * 100)
    f2 = open('generated/' + spinList.get() + '/RandomForest/lowFeatureList.txt', 'a+')
    f2.write(
        "Random Forest Accuracy Score -> " + str(round((accuracy_score(predictions_RF, Test_Ylow1) * 100), 2)) + "%\n")
    f2.write("Random Forest Confusion Matrix " + str(confusion_matrix(Test_Ylow1, predictions_RF)) + "\n")
    f2.write("Random Forest Classification Report " + str(classification_report(Test_Ylow1, predictions_RF)))
    f2.close()
    ''

def Logistic_Regression():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Train_Xlow1, Test_Xlow1, Train_Ylow1, Test_Ylow1, Train_X_Tfidflow1, Test_X_Tfidflow1, Corpus21, Corpuslow21
    try:
        if not os.path.exists("generated/"+spinList.get()+"/LogisticRegression"):
            os.makedirs("generated/"+spinList.get()+"/LogisticRegression")

    except OSError:
        print('Error: Creating Folder1')
    lr = LogisticRegression()
    lr.fit(Train_X_Tfidf1, Train_Y1)

    predictions_LR = lr.predict(Test_X_Tfidf1)
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
        if Corpus21['Spam_Ham'][Test_X1.index[i]] == 0:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_LR[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpus21['Spam_Ham'][Test_X1.index[i]] == 1:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_LR[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() + '/LogisticRegression/upFeatureList.csv', index=False)

    print("Logistic Regression Accuracy Score -> ", accuracy_score(predictions_LR, Test_Y1) * 100)
    f2 = open('generated/' + spinList.get() + '/LogisticRegression/upFeatureList.txt', 'a+')
    f2.write(
        "Logistic Regression Accuracy Score -> " + str(round((accuracy_score(predictions_LR, Test_Y1) * 100), 2)) + "%\n")
    f2.write("Logistic Regression Confusion Matrix " + str(confusion_matrix(Test_Y1, predictions_LR)) + "\n")
    f2.write("Logistic Regression Classification Report " + str(classification_report(Test_Y1, predictions_LR)))
    f2.close()

    # Random_Forest_model.fit(Train_X_Tfidflow1, Train_Ylow1, sample_weight=None)

    predictions_LR = lr.predict(Test_X_Tfidflow1)
    # Cross validation
    # accuracy = cross_validate(Random_Forest_model,Train_X_Tfidf,Train_Y,cv=10)['test_score']
    #
    # print('Random accuracy is: ',sum(accuracy)/len(accuracy)*100,'%')

    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_Xlow1)):
        if Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 0:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_LR[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 1:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_LR[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() + '/LogisticRegression/lowFeatureList.csv', index=False)

    print("Logistic Regression Accuracy Score -> ", accuracy_score(predictions_LR, Test_Ylow1) * 100)
    f2 = open('generated/' + spinList.get() + '/LogisticRegression/lowFeatureList.txt', 'a+')
    f2.write(
        "Logistic Regression Accuracy Score -> " + str(round((accuracy_score(predictions_LR, Test_Ylow1) * 100), 2)) + "%\n")
    f2.write("Logistic Regression Confusion Matrix " + str(confusion_matrix(Test_Ylow1, predictions_LR)) + "\n")
    f2.write("Logistic Regression Classification Report " + str(classification_report(Test_Ylow1, predictions_LR)))
    f2.close()

    ''

def Feed_ForwardNeural_Network():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Train_Xlow1, Test_Xlow1, Train_Ylow1, Test_Ylow1, Train_X_Tfidflow1, Test_X_Tfidflow1, Corpus21, Corpuslow21

    try:
        if not os.path.exists("generated/"+spinList.get()+"/FeedForwardNeuralNetwork"):
            os.makedirs("generated/"+spinList.get()+"/FeedForwardNeuralNetwork")

    except OSError:
        print('Error: Creating Folder1')

    from sklearn.neural_network import MLPClassifier

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

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
        if Corpus21['Spam_Ham'][Test_X1.index[i]] == 0:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_FFNN[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpus21['Spam_Ham'][Test_X1.index[i]] == 1:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_FFNN[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() + '/FeedForwardNeuralNetwork/upFeatureList.csv',
                                       index=False)

    print("FFNN Accuracy Score -> ", accuracy_score(predictions_FFNN, Test_Y1) * 100)
    f2 = open('generated/' + spinList.get() + '/FeedForwardNeuralNetwork/upFeatureList.txt', 'a+')
    f2.write(
        "FFNN Accuracy Score -> " + str(
            round((accuracy_score(predictions_FFNN, Test_Y1) * 100), 2)) + "%\n")
    f2.write("FFNN Confusion Matrix " + str(confusion_matrix(Test_Y1, predictions_FFNN)) + "\n")
    f2.write("FFNN Classification Report " + str(classification_report(Test_Y1, predictions_FFNN)))
    f2.close()

    # Random_Forest_model.fit(Train_X_Tfidflow1, Train_Ylow1, sample_weight=None)

    predictions_FFNN = clf.predict(Test_X_Tfidflow1)
    # Cross validation
    # accuracy = cross_validate(Random_Forest_model,Train_X_Tfidf,Train_Y,cv=10)['test_score']
    #
    # print('Random accuracy is: ',sum(accuracy)/len(accuracy)*100,'%')

    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_Xlow1)):
        if Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 0:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predictions_FFNN[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 1:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predictions_FFNN[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() + '/FeedForwardNeuralNetwork/lowFeatureList.csv',
                                       index=False)

    print("FFNN Accuracy Score -> ", accuracy_score(predictions_FFNN, Test_Ylow1) * 100)
    f2 = open('generated/' + spinList.get() + '/FeedForwardNeuralNetwork/lowFeatureList.txt', 'a+')
    f2.write(
        "FFNN Accuracy Score -> " + str(
            round((accuracy_score(predictions_FFNN, Test_Ylow1) * 100), 2)) + "%\n")
    f2.write("FFNN Confusion Matrix " + str(confusion_matrix(Test_Ylow1, predictions_FFNN)) + "\n")
    f2.write("FFNN Classification Report " + str(classification_report(Test_Ylow1, predictions_FFNN)))
    f2.close()



    ''

def K_nearest_neighbors():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Train_Xlow1, Test_Xlow1, Train_Ylow1, Test_Ylow1, Train_X_Tfidflow1, Test_X_Tfidflow1, Corpus21, Corpuslow21

    try:
        if not os.path.exists("generated/" + spinList.get() + "/KNearestNeighbors"):
            os.makedirs("generated/" + spinList.get() + "/KNearestNeighbors")

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


    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_X1)):
        if Corpus21['Spam_Ham'][Test_X1.index[i]] == 0:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpus21['Spam_Ham'][Test_X1.index[i]] == 1:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() + '/KNearestNeighbors/upFeatureList.csv',
                                       index=False)

    print("KNN Accuracy Score -> ", accuracy_score(predicted, Test_Y1) * 100)
    f2 = open('generated/' + spinList.get() + '/KNearestNeighbors/upFeatureList.txt', 'a+')
    f2.write(
        "KNN Accuracy Score -> " + str(
            round((accuracy_score(predicted, Test_Y1) * 100), 2)) + "%\n")
    f2.write("KNN Confusion Matrix " + str(confusion_matrix(Test_Y1, predicted)) + "\n")
    f2.write("KNN Classification Report " + str(classification_report(Test_Y1, predicted)))
    f2.close()

    # Random_Forest_model.fit(Train_X_Tfidflow1, Train_Ylow1, sample_weight=None)

    predicted = classifier.predict(Test_X_Tfidflow1)
    # Cross validation
    # accuracy = cross_validate(Random_Forest_model,Train_X_Tfidf,Train_Y,cv=10)['test_score']
    #
    # print('Random accuracy is: ',sum(accuracy)/len(accuracy)*100,'%')

    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_Xlow1)):
        if Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 0:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 1:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() + '/KNearestNeighbors/lowFeatureList.csv',
                                       index=False)

    print("KNN Accuracy Score -> ", accuracy_score(predicted, Test_Ylow1) * 100)
    f2 = open('generated/' + spinList.get() + '/KNearestNeighbors/lowFeatureList.txt', 'a+')
    f2.write(
        "KNN Accuracy Score -> " + str(
            round((accuracy_score(predicted, Test_Ylow1) * 100), 2)) + "%\n")
    f2.write("KNN Confusion Matrix " + str(confusion_matrix(Test_Ylow1, predicted)) + "\n")
    f2.write("KNN Classification Report " + str(classification_report(Test_Ylow1, predicted)))
    f2.close()



    ''

def Radial_Basis():
    global Train_X1, Test_X1, Train_Y1, Test_Y1, Train_X_Tfidf1, Test_X_Tfidf1, Train_Xlow1, Test_Xlow1, Train_Ylow1, Test_Ylow1, Train_X_Tfidflow1, Test_X_Tfidflow1, Corpus21, Corpuslow21

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





    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_X1)):
        if Corpus21['Spam_Ham'][Test_X1.index[i]] == 0:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpus21['Spam_Ham'][Test_X1.index[i]] == 1:
            model_results['Message'].append(Corpus21['upFeatureList'][Test_X1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpus21['Spam_Ham'][Test_X1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() + '/RadialBasis/upFeatureList.csv',
                                       index=False)

    print("RB Accuracy Score -> ", accuracy_score(predicted, Test_Y1) * 100)
    f2 = open('generated/' + spinList.get() + '/RadialBasis/upFeatureList.txt', 'a+')
    f2.write(
        "RB Accuracy Score -> " + str(
            round((accuracy_score(predicted, Test_Y1) * 100), 2)) + "%\n")
    f2.write("RB Confusion Matrix " + str(confusion_matrix(Test_Y1, predicted)) + "\n")
    f2.write("RB Classification Report " + str(classification_report(Test_Y1, predicted)))
    f2.close()

    # Random_Forest_model.fit(Train_X_Tfidflow1, Train_Ylow1, sample_weight=None)

    predicted = clf.predict(Test_X_Tfidflow1)
    # Cross validation
    # accuracy = cross_validate(Random_Forest_model,Train_X_Tfidf,Train_Y,cv=10)['test_score']
    #
    # print('Random accuracy is: ',sum(accuracy)/len(accuracy)*100,'%')

    model_results = {'Message': [],
                     'Original_Spam_Ham': [],
                     'Predicted': [],
                     'Spam_Ham': []
                     }
    for i in range(len(Test_Xlow1)):
        if Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 0:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('0')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])

            # print(Corpus2['upFeatureList'][Test_X.index[i]],'  0   ',Corpus21['Spam_Ham'][Test_X.index[i]])
        elif Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]] == 1:
            model_results['Message'].append(Corpuslow21['lowFeatureList'][Test_Xlow1.index[i]])
            model_results['Original_Spam_Ham'].append('1')
            model_results['Predicted'].append(predicted[i])
            model_results['Spam_Ham'].append(Corpuslow21['Spam_Ham'][Test_Xlow1.index[i]])
            # print(Corpus2['v2'][Test_X.index[i]], '  1   ', predicted[i], '     ', Corpus2['v1'][Test_X.index[i]])
        # i = i + 1

    pd.DataFrame(model_results).to_csv('generated/' + spinList.get() + '/RadialBasis/lowFeatureList.csv',
                                       index=False)

    print("RB Accuracy Score -> ", accuracy_score(predicted, Test_Ylow1) * 100)
    f2 = open('generated/' + spinList.get() + '/RadialBasis/lowFeatureList.txt', 'a+')
    f2.write(
        "RB Accuracy Score -> " + str(
            round((accuracy_score(predicted, Test_Ylow1) * 100), 2)) + "%\n")
    f2.write("RB Confusion Matrix " + str(confusion_matrix(Test_Ylow1, predicted)) + "\n")
    f2.write("RB Classification Report " + str(classification_report(Test_Ylow1, predicted)))
    f2.close()
    ''




########################################################################################################################
########################################################################################################################
#Define a function to perform the 1D-TP feature extraction on SMS
addAllGetParttern = []

upFeatureListAll = []
lowFeaturesListAll = []
# global originalSMS1
def perform1DTPOnSMS(x1, x2, ii, nn, mm, originalSMS):
    import binaryToDecimal
    # global originalSMS1
    # print(originalSMS1)
    # print("The given sms: ", originalSMS, " with length: ", len(originalSMS))

    # fwritersms = open("generated/sms/optimization/sms.txt", "a+")
    # fwritersms.write("The given sms: "+ originalSMS+ " with length: "+ str(len(originalSMS))+'\n')



    SMS_length = len(originalSMS)
    # Set the initial value of P
    P = x2
    # Get the number of possible pattern P that can be formed on the SMS
    noOfPatterns = SMS_length // (P + 1)
    # print("The number of pattern is: ", noOfPatterns, " and value of ρ is: ", P)
    aa = 'The number of pattern is: '+ str(noOfPatterns), ' and value of ρ is: '+ str(P)
    # fwritersms.write(str(''.join(aa).encode('utf-8')))
    # fwritersms.close()


    addAllGetParttern.append(aa)
    upFeaturesList, lowFeaturesList = [], []
    n, step = 1, P + 1


#     fwriterupper = open("generated/upper/optimization/upper.txt", "a+")
#     fwriterupperFeature = open("generated/upperFeature/optimization/upperFeature.txt", "a+")
#
#     fwriterlower = open("generated/lower/optimization/lower.txt", "a+")
#     fwriterlowerFeature = open("generated/lowerFeature/optimization/lowerFeature.txt", "a+")
#
#
# ######################################
#     fwriterleft = open("generated/left/optimization/left.txt", "a+")
#     fwritercenter = open("generated/center/optimization/center.txt", "a+")
#
#     fwriterright = open("generated/right/optimization/right.txt", "a+")
    # fwriterlowerFeature = open("generated/lowerFeature/lowerFeature.txt", "a+")


    for k in range(0, (noOfPatterns * step), step):
        text = originalSMS[k:n * step]
        aa = "Pattern "+ str(n)+" Text is: "+ str(text)

        # fwriterupper.write(aa+'\n')
        # fwriterupperFeature.write(aa+'\n')
        # fwriterlower.write(aa+'\n')
        # fwriterlowerFeature.write(aa+'\n')
        # fwriterleft.write(aa+'\n')
        # fwritercenter.write(aa+'\n')
        # fwriterright.write(aa+'\n')




        addAllGetParttern.append(aa)
        # print("Pattern ", n, " Text is: ", text)
        # Partition the text into 3 list(Pl, Pr, Pc)
        Lp, Cp, Rp = text[:P // 2], text[P // 2:P // 2 + 1], text[P // 2 + 1:]
        # print("The left list is: ", Lp)
        aa = "The left list is: "+ str(Lp)
        # fwriterleft.write(aa+'\n')


        addAllGetParttern.append(aa)
        # print("The centre list is: ", Cp)
        aa = "The centre list is: "+str(Cp)

        # fwritercenter.write(aa+'\n')


        addAllGetParttern.append(aa)
        # print("The right list is: ", Rp)
        aa = "The right list is: "+str(Rp)

        # fwriterright.write(aa+'\n')

        addAllGetParttern.append(aa)
        # Get the UTF-8 of the characters in the given SMS text pattern
        Pl, Pr, Pc = [], [], ord(Cp)
        for i in range(0, len(Lp)):
            Pl.insert(i, ord(Lp[i]))
        for i in range(0, len(Rp)):
            Pr.insert(i, ord(Rp[i]))
        # print("The left list is: ", Pl)
        aa = "The left list is: "+ str(Pl)
        # fwriterleft.write(aa+'\n')

        addAllGetParttern.append(aa)
        # print("The centre list is: ", Pc)
        aa = "The centre list is: "+str(Pc)

        # fwritercenter.write(aa+'\n')


        addAllGetParttern.append(aa)
        # print("The right list is: ", Pr)
        aa = "The right list is: "+str(Pr)

        # fwriterright.write(aa+'\n')

        addAllGetParttern.append(aa)
        # Set the threshold value and perform the 1D-TP transformation
        B = x1
        # print("The value of β is: ", B)
        aa = "The value of β is: "+str(B)

        # fwriterupper.write(str(''.join(aa).encode('utf-8'))+'\n')
        # fwriterupperFeature.write(str(''.join(aa).encode('utf-8'))+'\n')
        # fwriterlower.write(str(''.join(aa).encode('utf-8'))+'\n')
        # fwriterlowerFeature.write(str(''.join(aa).encode('utf-8'))+'\n')
        # fwriterleft.write(str(''.join(aa).encode('utf-8'))+'\n')
        # fwritercenter.write(str(''.join(aa).encode('utf-8'))+'\n')
        # fwriterright.write(str(''.join(aa).encode('utf-8'))+'\n')
        addAllGetParttern.append(str(''.join(aa).encode('utf-8'))+'\n')
        # Comparison of Pc with neighbors (Pi),
        Tpl, Tpr = [], []
        for i in range(0, len(Lp)):
            if (Pc > (Pl[i] + B)):
                Tpl.insert(i, 1)
            elif (Pc <= (Pl[i] + B) and Pc >= (Pl[i] - B)):
                Tpl.insert(i, 0)
            elif (Pc < (Pl[i] - B)):
                Tpl.insert(i, -1)
        for i in range(0, len(Rp)):
            if (Pc > (Pr[i] + B)):
                Tpr.insert(i, 1)
            elif (Pc <= (Pr[i] + B) and Pc >= (Pr[i] - B)):
                Tpr.insert(i, 0)
            elif (Pc < (Pr[i] - B)):
                Tpr.insert(i, -1)
        # print("The left list is: ", Tpl)
        aa = "The left list is: "+str(Tpl)

        # fwriterleft.write(aa+'\n')

        addAllGetParttern.append(aa)
        # print("The right list is: ", Tpr)
        aa = "The right list is: "+str(Tpr)

        # fwriterright.write(aa+'\n')

        addAllGetParttern.append(aa)
        # Separation positive and negative values,
        upF, lowF = [], []
        for i in range(0, len(Lp)):
            if (Tpl[i] == -1):
                upF.insert(i, 0)
                lowF.insert(i, 1)
            else:
                upF.insert(i, Tpl[i])
                lowF.insert(i, 0)
        for i in range(0, len(Rp)):
            if (Tpr[i] == -1):
                upF.insert(len(Lp) + i, 0)
                lowF.insert(len(Lp) + i, 1)
            else:
                upF.insert(len(Lp) + i, Tpr[i])
                lowF.insert(len(Lp) + i, 0)
        # print("The up list is: ", upF)
        aa = "The up list is: "+str(upF)

        # fwriterupper.write(aa+'\n')


        addAllGetParttern.append(aa)
        # print("The low list is: ", lowF)
        aa = "The low list is: "+str(lowF)

        # fwriterlower.write(aa+'\n')


        addAllGetParttern.append(aa)
        # Conversion of binary values to decimal
        upFeatures = binaryToDecimal.binaryToDecimal(upF)
        lowFeatures = binaryToDecimal.binaryToDecimal(lowF)
        # print("The upFeatures is: ", upFeatures)
        aa = "The upFeatures is: "+str(upFeatures)

        # fwriterupperFeature.write(aa+'\n')

        addAllGetParttern.append(aa)
        # print("The lowFeatures is: ", lowFeatures)
        aa = "The lowFeatures is: "+str(lowFeatures)

        # fwriterlowerFeature.write(aa+'\n')


        addAllGetParttern.append(aa)
        # Populate the both upFeatures List and lowFeaturesList
        upFeaturesList.insert(n - 1, upFeatures)
        lowFeaturesList.insert(n - 1, lowFeatures)
        # increase the value of n for the pattern to be selected for computation
        n += 1


    # print("The upFeatures List is: ", upFeaturesList)   and ii == 1
    aa = "The upFeatures List is: "+str(upFeaturesList)
    if mm == 2:
        # print('BBBBBBBBBBBB SSSSSSSSS ',ii,'  ',nn)
        upFeatureListAll.append(upFeaturesList)
        lowFeaturesListAll.append(lowFeaturesList)
    # fwriterupperFeature.write(aa+'\n')


    addAllGetParttern.append(aa)
    # print("The lowFeatures List is: ", lowFeaturesList)
    aa = "The lowFeatures List is: "+str(lowFeaturesList)

    # fwriterlowerFeature.write(aa+'\n')

    addAllGetParttern.append(aa)
    avg = 0.0
    # fileList = Listbox(main2, height=18, width=40)
    # i = 0
    # for m in addAllGetParttern:
    #     i += 1
    #     fileList.insert(i, str(m).replace("'", "").replace(",", "").replace("(", "").replace(")", ""))
    #

    # fwriterupper.close()
    # fwriterupperFeature.close()
    # fwriterlower.close()
    # fwriterlowerFeature.close()
    # fwriterleft.close()
    # fwritercenter.close()
    # fwriterright.close()



    # fileList.place(x=10, y=290)
    if (len(upFeaturesList) != 0):
        avg = sum(upFeaturesList) / len(upFeaturesList)
    return avg

#============================== OPTIMIZATION PART USING SIMULATED ANNEALING ALGORITHM ========================================================
# define the objective function
def f(x, i, n, m, originalSMS):
	x1 = x[0]
	x2 = x[1]
	if(x2 < 6):
		x2 = abs(x2) + 6
	obj = perform1DTPOnSMS(int(abs(x1)), int(abs(x2)), i, n, m, originalSMS)
	return obj

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

extractButton = Button(mainDisplay, text="Logistic Regression", height=1, command=Logistic_Regression)
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
    DT = open(r'generated/'+spinList.get()+'/LogisticRegression/DecisionTreeResult.txt').readlines()
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

