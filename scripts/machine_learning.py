import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import keras
from keras.models import Sequential
from keras.utils import pad_sequences, to_categorical
from keras import layers
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc

max_words = 5000
max_len = 200


def main():
    # load data into a pandas DataFrame
    data = pd.read_csv('./data/result.csv', sep=',')

    check_nan(data)
    data['comment'] = data['comment'].astype('string')
    data['label'] = data['label'].astype('int')
    print(data.info())
    print(data.sample(5))

    # split data into training and testing sets
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data.comment)
    sequences = tokenizer.texts_to_sequences(data.comment)
    war = pad_sequences(sequences, maxlen=200)

    labels = to_categorical(data['label'], num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(
        war, labels, test_size=0.1, stratify=labels, random_state=42)

    # ETAP 2: CLASSIC ML
    show_plot(data)

    # ETAP 3: NEURAL MODEL
    fine_tune(X_train, X_test, y_train, y_test)

    best_model = keras.models.load_model('./results/best_model.h5')
    test_loss, test_acc = best_model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    # ETAP 4: LANGUAGE MODEL
    finbert = BertForSequenceClassification.from_pretrained(
        'yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

    sentences = ['idk man sounds like brainwashed fanatic',
                 'there is a shortage of capital, and we need extra financing.',
                 'formulation patents might protect Vasotec to a limited extent.']

    nlp = pipeline('text-classification',
                   model=finbert,
                   tokenizer=tokenizer)

    results = nlp(sentences)
    # What next??
    print(results)


def fine_tune(X_train, X_test, y_train, y_test):
    best_val_loss = float('inf')
    best_model = None
    best_history = None

    for epoch in range(10):
        model = modelPlot()
        history = model.fit(X_train, y_train, epochs=10,
                            validation_data=(X_test, y_test))
        val_loss = model.evaluate(X_test, y_test)[0]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_history = history
            best_model.save('./results/best_model.h5')
    show_history(best_history)
    plot_history(best_history)
    plt.close()


def modelPlot():
    models = Sequential()

    models.add(layers.Embedding(max_words, 40, input_length=max_len))
    models.add(layers.Conv1D(20, 6, activation='relu'))
    models.add(layers.MaxPooling1D(5))
    models.add(layers.Conv1D(20, 6, activation='relu'))
    models.add(layers.GlobalMaxPooling1D())
    models.add(layers.Dense(3, activation='softmax'))

    models.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    return models


# ETAP 2: CLASSIC ML
# Choose 3 models to fit data and present the results with
# confusion matric and roc curve.
def show_plot(data):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['comment'], data['label'], test_size=0.3, random_state=42)

    tfidf = TfidfVectorizer(max_features=500000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    models = [
        ('Multinomial Naive Bayes', MultinomialNB()),
        ('Logistic Regression', LogisticRegression()),
        ('Random Forest', RandomForestClassifier())
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        print(name)
        print('Confusion Matrix:\n', cm)
        print('ROC AUC:', roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
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
