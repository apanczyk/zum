import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import type_of_target


def main():
    # load data into a pandas DataFrame
    data = pd.read_csv('result.csv', sep=',')

    check_nan(data)
    data['comment'] = data['comment'].astype('string')
    data['label'] = data['label'].astype('int')
    print(data.info())

    # split data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(
        data['comment'], data['label'], test_size=0.1)

    # vectorize comments using tf-idf
    vectorizer = TfidfVectorizer(max_features=500000, ngram_range=(1, 2))
    print(X_train)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    show_plot(X_train, X_test, y_train, y_test)


# Choose 3 models to fit data and present the results with
# confusion matric and roc curve.
def show_plot(X_train, X_test, y_train, y_test):
    models = [
        ('Multinomial Naive Bayes', MultinomialNB()),
        ('Logistic Regression', LogisticRegression()),
        ('Random Forest', RandomForestClassifier())
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # roc_auc = auc(fpr, tpr)

        print(name)
        print('Confusion Matrix:\n', cm)
        # print('ROC AUC:', roc_auc)

        plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve: {}'.format(name))
        plt.legend(loc="lower right")
        plt.show()


def check_nan(data):
    i = 0
    for val in data['comment']:
        i += 1
        if val is np.nan:
            print(f"REMOVE {i}")


main()
