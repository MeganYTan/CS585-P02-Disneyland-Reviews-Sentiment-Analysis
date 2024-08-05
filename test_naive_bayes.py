import sys
import pandas as pd
import numpy as np
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import lil_matrix
from scipy.sparse import vstack
from sklearn.metrics import confusion_matrix
import seaborn as sns
import math


def clean_data():
    # lower case, remove stopwords, get sentiments
    # helper functions to clean data
    def sentiment(score):
        if score == "HAM":
            return 1
        else:
            return 0

    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        text = text.lower()
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return ' '.join(filtered_text)

    df.drop_duplicates(keep='first', inplace=True)
    df['Sentiment'] = df['Category'].apply(sentiment)
    df['Review_Text_No_Stopwords'] = df['Sentence'].apply(remove_stopwords)


# index to get column numbers
def create_sparse_binary_bag_of_words(text):
    # init sparse matrix with length of vocabulary
    bag_of_words_sparse = lil_matrix((1, len(vocabulary_index)))
    words = word_tokenize(text)
    for word in words:
        if word in vocabulary_index:
            bag_of_words_sparse[0, vocabulary_index[word]] = 1
    return bag_of_words_sparse


def get_vocabulary_index_from_df():
    # vocab is set of ALL WORDS
    vocabulary = df['Review_Text_No_Stopwords'].str.split(expand=True).stack().unique().tolist()
    vocabulary = [word.lower() for word in vocabulary]
    vocabulary_index = {word: i for i, word in enumerate(vocabulary)}
    return vocabulary_index


def get_training_and_test_sets():
    # training set depends on TRAIN_SIZE_P
    training_set = df.iloc[:int(len(df) * TRAIN_SIZE_P)]
    # test set is last 20%
    test_set = df.iloc[int(len(df) * 0.8):]
    return training_set, test_set


def get_sparse_matrix_X(dataframe):
    # x is 'Review_Text_No_Stopwords
    sparse_matrix = dataframe['Review_Text_No_Stopwords'].apply(lambda x: create_sparse_binary_bag_of_words(x))
    # combine to single sparse matrix
    X = vstack(sparse_matrix)
    return X


def get_X_train():
    return get_sparse_matrix_X(training_set)


def get_X_test():
    return get_sparse_matrix_X(test_set)


def calculate_probabilities_in_training_set():
    num_rows = len(training_set)
    num_rows_positive = (y_train == 1).sum()
    num_rows_negative = (y_train == 0).sum()
    probability_positive = num_rows_positive / num_rows
    probability_negative = num_rows_negative / num_rows
    return probability_positive, probability_negative


def get_word_count_by_y():
    training_set['Word_Count'] = training_set['Review_Text_No_Stopwords'].apply(lambda x: len(word_tokenize(x)))
    word_count_by_y = training_set.groupby('Sentiment')['Word_Count'].sum()
    return word_count_by_y


def calculate_likelihoods():
    X_train_coo = X_train.tocoo()
    feature_counts_positive = np.zeros(X_train.shape[1])
    feature_counts_negative = np.zeros(X_train.shape[1])

    for i, j, v in zip(X_train_coo.row, X_train_coo.col, X_train_coo.data):
        if y_train[i] == 0:
            feature_counts_negative[j] += v
        else:
            feature_counts_positive[j] += v
    # for each word and label get probability =  ((count x = X and y = Y)+1) / (count(number of words when y = Y) + # of words in Vocabulary)
    feature_counts_negative = feature_counts_negative + 1
    feature_counts_positive = feature_counts_positive + 1
    likelihoods_negative = (feature_counts_negative) / (word_count_by_y[0] + len(vocabulary_index))
    likelihoods_positive = (feature_counts_positive) / (word_count_by_y[1] + len(vocabulary_index))
    # each number is probability(x = word<corresponding to index of col in X_train> | y = positive/negative)
    return likelihoods_positive, likelihoods_negative


def round_to_significant_digits(num, sig_digits):
    if num == 0 or math.isnan(num):
        return 0
    else:
        return round(num, sig_digits - int(math.floor(math.log10(abs(num)))) - 1)



def predict(X, probability_negative, probability_positive, likelihoods_negative, likelihoods_positive):
    X_coo = X.tocoo()
    print(X_coo)
    predictions = np.zeros(X.shape[0])
    # use log as per instructions
    log_probability_negative = np.log(probability_negative)
    log_probability_positive = np.log(probability_positive)
    log_likelihoods_negative = np.log(likelihoods_negative)
    log_likelihoods_positive = np.log(likelihoods_positive)
    print(likelihoods_positive)
    print(likelihoods_negative)
    print(log_likelihoods_positive)
    print(log_likelihoods_negative)
    # for each document
    for i in range(X.shape[0]):
        log_posterior_probability_negative = log_probability_negative
        log_posterior_probability_positive = log_probability_positive

        for j in range(X_coo.nnz):
            if X_coo.row[j] == i:
                feature_index = X_coo.col[j]
                log_posterior_probability_negative += log_likelihoods_negative[feature_index]
                log_posterior_probability_positive += log_likelihoods_positive[feature_index]


        predictions[i] = 1 if log_posterior_probability_positive > log_posterior_probability_negative else 0
    # this is for when predicting user sentence
    posterior_probability_positive = round_to_significant_digits(np.exp(log_posterior_probability_positive), 5)
    posterior_probability_negative = round_to_significant_digits(np.exp(log_posterior_probability_negative), 5)
    return predictions, posterior_probability_positive, posterior_probability_negative


def print_metrics(y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm)
    TP, FP = cm[0]
    FN, TN = cm[1]

    # Calculate metrics
    sensitivity = round_to_significant_digits(TP / (TP + FN), 5)
    specificity = round_to_significant_digits(TN / (TN + FP), 5)
    precision = round_to_significant_digits(TP / (TP + FP), 5)
    npv = round_to_significant_digits(TN / (TN + FN), 5)
    accuracy = round_to_significant_digits((TP + TN) / (TP + FP + FN + TN), 5)
    f_score = round_to_significant_digits(2 * (precision * sensitivity) / (precision + sensitivity), 5)

    print(f'''Test results / metrics:

    Number of true positives: {TP}
    Number of true negatives: {TN}
    Number of false positives: {FP}
    Number of false negatives: {FN}
    Sensitivity (recall): {sensitivity}
    Specificity: {specificity}
    Precision: {precision}
    Negative predictive value: {npv}
    Accuracy: {accuracy}
    F-score: {f_score}
    ''')


def plot_cm(cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def get_and_classify_user_input():
    def vectorize_user_input():
        user_input_data = {user_input}
        user_input_df = pd.DataFrame(user_input_data)
        sparse_matrix = user_input_df[0].apply(lambda x: create_sparse_binary_bag_of_words(x))
        user_input_s_array = vstack(sparse_matrix)
        return user_input_s_array

    user_input = input("Please enter your sentence: ")
    user_input_s_array = vectorize_user_input()
    print(user_input_s_array)
    prediction, posterior_probability_positive, posterior_probability_negative = predict(user_input_s_array,
                                                                                         probability_negative,
                                                                                         probability_positive,
                                                                                         likelihoods_negative,
                                                                                         likelihoods_positive)
    label = "POSITIVE" if prediction[0] == 1 else "NEGATIVE"
    print(f'''Sentence S:

     {user_input}

    was classified as {label}.
    P(POSITIVE | S) = {posterior_probability_positive}
    P(NEGATIVE | S) = {posterior_probability_negative}''')


if __name__ == '__main__':
    # size in percentage is 80 by default
    TRAIN_SIZE = 80

    # if ran from command line
    if sys.stdin.isatty():
        cl_arguments = sys.argv[1:]

        if len(cl_arguments) == 1:
            arg = cl_arguments[0]
            if type(arg) == "number" and int(arg):
                TRAIN_SIZE = int(arg)

    TRAIN_SIZE_P = TRAIN_SIZE / 100
    nltk.download('punkt')
    nltk.download('stopwords')

    print(f'''Tan, Megan, A20527707 solution:
    Training set size: {TRAIN_SIZE}%''')

    # step 0 - set up

    # step 1 read and clean data
    df = pd.read_csv("test.csv")
    clean_data()

    # Step 2 training classifier
    print("Training classifier...")
    # step 2.2 split into training and test sets
    training_set, test_set = get_training_and_test_sets()
    # step 2.1 get vocab from ALL words in data set
    vocabulary_index = get_vocabulary_index_from_df()
    # get X train
    X_train = get_X_train()
    # get Y Train
    y_train = training_set['Sentiment']
    y_train = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train
    # use bag of words
    X_test = get_X_test()
    y_test = test_set['Sentiment']
    # step 2.3 use binary BOG with add 1 smoothing
    probability_positive, probability_negative = calculate_probabilities_in_training_set()
    word_count_by_y = get_word_count_by_y()
    likelihoods_positive, likelihoods_negative = calculate_likelihoods()

    # step 3 testing classifier
    print("Testing Classifier...")
    # step 3.1 test classifier
    y_pred, posterior_probability_positive, posterior_probability_negative = predict(X_test, probability_negative,
                                                                                     probability_positive,
                                                                                     likelihoods_negative,
                                                                                     likelihoods_positive)
    # step 3.2 print metrics
    print_metrics(y_pred, y_test)

    while True:
        get_and_classify_user_input()
        should_continue = input("Do you want to enter another sentence? [Y/N] ")
        if should_continue.upper() != "Y":
            quit()
