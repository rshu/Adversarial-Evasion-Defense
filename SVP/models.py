from readData import read
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

dataset = "phpmyadmin"

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = read(dataset)
    # print(X_train)
    # print(X_test)
    models = [RandomForestClassifier(),
              DecisionTreeClassifier(),
              SVC(probability=True, verbose= False),
              LogisticRegression(max_iter=4000,verbose=False),
              MLPClassifier(max_iter=4000)]
    for model in models:
        print("----------model----------")
        print(model)
        print("Fitting ...")
        model.fit(X_train, y_train)

        print("Evaluating ...")
        y_pred = model.predict(X_test)

        print("Accuracy is %f." % accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print("Precision score is %f." % precision_score(y_test, y_pred))
        print("Recall score is %f." % recall_score(y_test, y_pred))
        print("F1 score is %f." % f1_score(y_test, y_pred))

        # probability=True should be set, default is false
        probs = model.predict_proba(X_test)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
