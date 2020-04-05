import read
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

TUNING = False  # Set this to False if you don't want to tune

print("Reading data ...")
x_all, y_all = read.read(load_data=False)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

if TUNING:

    tuned_parameters = [{'penalty': ['l2'], 'solver': ["lbfgs", "sag"], 'C': [0.01, 0.1, 1, 10, 100]},
                        {'penalty': ['l1'], 'solver': ["liblinear", "saga"], 'C': [0.01, 0.1, 1, 10, 100]}]

    scores = ["accuracy", "f1"]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print("")

        clf = GridSearchCV(linear_model.LogisticRegression(), tuned_parameters, cv=5, scoring=score)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print("")
        print(clf.best_estimator_)
        print(clf.best_params_)
        print("")
        print("Grid scores on development set:")
        print("")
        print(clf.best_score_)
        # for params, mean_score, scores in clf.cv_results_:
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean_score, scores.std() / 2, params))
        print("")

        print("Detailed classification report:")
        print("")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print("")
        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred))
        print("")

else:
    models = [
        linear_model.LogisticRegression(),
        linear_model.LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                                        intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                        penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
                                        verbose=0, warm_start=False),
        linear_model.LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                                        intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                        penalty='l2', random_state=None, solver='lbfgs', tol=0.0001,
                                        verbose=0, warm_start=False)
    ]
    for model in models:
        print("----------model----------")
        print(model)
        print("Fitting logistic regression ...")
        model.fit(x_train, y_train)

        print("Evaluating ...")
        y_pred = model.predict(x_test)

        print("Accuracy is %f." % accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print("Precision score is %f." % precision_score(y_test, y_pred))
        print("Recall score is %f." % recall_score(y_test, y_pred))
        print("F1 score is %f." % f1_score(y_test, y_pred))


