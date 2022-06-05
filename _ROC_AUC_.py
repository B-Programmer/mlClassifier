import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

#import dataset from CSV file on Github
# url = 'generated1\without optimization\British English SMS Corpora\0\FeedForwardNeuralNetwork\upFeatureList.csv'
aa = ['SMS Spam Corpus v.0.1', 'British English SMS Corpora', 'Kaggle']
bb = ['FeedForwardNeuralNetwork', 'KNearestNeighbors',
      'LogisticRegression', 'Naive', 'RadialBasis',
      'RandomForest', 'svm']
for aaa in aa:
    model_results = {'Dataset': [],
                     'Alagorithm': [],
                     'Type': [],
                     'Part': [],
                     '%': [],
                     'Correct_Ham': [],
                     'Incorrect_Ham': [],
                     'Correct_Spam': [],
                     'Incorrect_Spam': []
                     }
    for bbb in bb:
        for dd in range(0, 5):
            # print(dd)
            # file = 'generated1\with optimization\SMS Spam Corpus v.0.1\+str(dd)+'\SVM\upFeature.csv'  , 'upFeatureList'
            cccc = ['lowFeatureList', 'upFeatureList']

            for cc in cccc:

                strings = ['generated1', 'with optimization', aaa, str(dd), bbb, cc+'.csv', ]#out
                file = '\\'.join(strings)
                print(file)
                data = pd.read_csv(r''+file)

                #define the predictor variables and the response variable
                X = data[['Original_Spam_Ham']]
                y = data['Predicted']
                #
                # #split the dataset into training (70%) and testing (30%) sets
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=3)


                #instantiate the model
                log_regression = LogisticRegression()

                #fit the model using the training data
                # print(X_train)
                log_regression.fit(X_train,y_train)

                #define metrics
                y_pred_proba = log_regression.predict_proba(X_test)[::,1]
                fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

                tn, fp, fn, tp = confusion_matrix(X, y).ravel()

                # print(y_test)
                y_pre =  log_regression.predict(X_test)

                correct_ham = 0
                incorrect_ham = 0
                correct_spam = 0
                incorrect_spam = 0

                for i in range(len(X_test)):
                    # print(y_pre[i], 'Lexxxx')
                    # print(y_test[X_test.index[i]], 'BLexxxx')

                    if y_pre[i] == 0 and y_test[X_test.index[i]] == 0:
                        correct_ham += 1
                        ''
                    elif y_pre[i] == 0 and y_test[X_test.index[i]] == 1:
                        incorrect_ham += 1
                        ''
                    elif y_pre[i] == 1 and y_test[X_test.index[i]] == 1:
                        correct_spam += 1
                        ''
                    elif y_pre[i] == 1 and y_test[X_test.index[i]] == 0:
                        incorrect_spam += 1
                        ''
                    # if data['all'][X_test.index[i]] == 0:
                    #
                    #     ''
                    # elif data['all'][X_test.index[i]] == 0:
                    #
                    #     ''


                #create ROC curve
                # plt.plot(fpr,tpr, color='orange', label='ROC')
                # plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
                # plt.ylabel('True Positive Rate')
                # plt.xlabel('False Positive Rate')
                #
                # plt.title('Receiver Operating Characteristic (ROC) Curve')
                # plt.legend()
                # plt.savefig('./generate_new/'+aaa+'_'+ bbb+'_'+cc +'_'+str(dd)+'_ROC_Curve.png', dpi=300, bbox_inches='tight')
                #
                # plt.show()

                #define metrics
                # y_pred_proba = log_regression.predict_proba(X_test)[::,1]
                # fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        #         auc = metrics.roc_auc_score(y_test, y_pred_proba)
        #
        #         #create ROC curve
        #
        #
        #         # sensitivity = tp / (tp + fn)
        #         # specificity = tn / (tn + fp)
        #         # precision  = tp / (tp + fp)
        #         #
        #         # print(sensitivity)
        #         # print(specificity)
        #         # print(precision)
        #
                accuracy = metrics.accuracy_score(X, y)
        #         print('Accuracy ', accuracy)
        #         precision_positive = metrics.precision_score(X, y, pos_label=1)
        #         print('precision_positive ', precision_positive)
        #         precision_negative = metrics.precision_score(X, y, pos_label=0)
        #         print('precision_negative ', precision_negative)
        #         recall_sensitivity = metrics.recall_score(X, y, pos_label=1)
        #         print('recall_sensitivity ', recall_sensitivity)
        #         recall_specificity = metrics.recall_score(X, y, pos_label=0)
        #         print('recall_specificity ', recall_specificity)
        #         f1_positive = metrics.f1_score(X, y, pos_label=1)
        #         print('f1_positive ', f1_positive)
        #         f1_negative = metrics.f1_score(X, y, pos_label=0)
        #         print('f1_negative ', f1_negative)
        #
                model_results['Dataset'].append(aaa)
                model_results['Alagorithm'].append(bbb)
                model_results['Type'].append(dd)
                model_results['Part'].append(cc)
                model_results['%'].append(str(accuracy))
                model_results['Correct_Ham'].append(str(correct_ham))
                model_results['Incorrect_Ham'].append(str(incorrect_ham))
                model_results['Correct_Spam'].append(str(correct_spam))
                model_results['Incorrect_Spam'].append(str(incorrect_spam))
        #         model_results['f1_positive'].append(str(f1_positive))
        #         model_results['f1_negative'].append(str(f1_negative))
        #
        #
        #
        #         # xx = recall_sensitivity
        #         # yy = recall_specificity
        #         #
        #         # # fig = plt.figure(figsize=(5,5))
        #         # # ax1 = fig.add_subplot(111)
        #         #
        #         # plt.plot(xx,yy, label="", color='brown')
        #         # plt.grid(True, which='both')
        #         # plt.xlabel('Sensitivity (proportion of breaching\nperiods predicted correctly)')
        #         # plt.ylabel('Specificity (proportion of non-breaching\nperiods predicted correctly)')
        #         #
        #         #
        #         #
        #         # plt.savefig('SVM_4_Sensitivity_Specificity.png', dpi=300, bbox_inches='tight')
        #         # plt.show()
        #
        #         plt.plot(fpr,tpr,label="AUC="+str(auc), color='green')
        #         plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        #         plt.ylabel('True Positive Rate')
        #         plt.xlabel('False Positive Rate')
        #         plt.legend(loc=4)
        #         plt.savefig('./generate_new/'+aaa+'_'+ bbb+'_'+cc   +'_'+str(dd)+'_ROC_Curve_AUC.png', dpi=300, bbox_inches='tight')
        #         # plt.show()
        #         plt.draw()
        #         plt.pause(0.0001)
        #         plt.clf()
        #
    pd.DataFrame(model_results).to_csv('./generate_new/'+  aaa+'.csv', index=False)