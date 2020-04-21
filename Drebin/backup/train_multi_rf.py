from backup import read
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

TUNING = False  # Set this to False if you don't want to tune

print("Reading data ...")
x_all, y_all = read.read_multiclass(load_data=False)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

if TUNING:

    tuned_parameters = [{'n_estimators': [10, 100, 1000],
                         'max_features': ["auto", "sqrt", "log2", None]}]

    scores = ["accuracy"]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print("")

        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring=score, n_jobs=2)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print("")
        print(clf.best_estimator_)
        print(clf.best_params_)
        print("")
        print("Grid scores on development set:")
        print("")
        print(clf.best_score_)
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
    models = [RandomForestClassifier(),
              RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                     max_depth=None, max_features='log2', max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, min_impurity_split=None,
                                     min_samples_leaf=1, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
                                     oob_score=False, random_state=None, verbose=0,
                                     warm_start=False)

              ]
    for model in models:
        print("Fitting RF ...")
        model.fit(x_train, y_train)

        print("Evaluating ...")
        y_pred = model.predict(x_test)

        print("Accuracy is %f." % accuracy_score(y_test, y_pred))
        print("-----------------------------------")


