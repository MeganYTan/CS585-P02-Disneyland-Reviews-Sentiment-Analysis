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
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def clean_data():
    # lower case, remove stopwords, get sentiments
    # helper functions to clean data
    def sentiment(score):
        if score > 3:
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
    df['Sentiment'] = df['Rating'].apply(sentiment)
    df['Review_Text_No_Stopwords'] = df['Review_Text'].apply(remove_stopwords)


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


def undersample_training_set():
    minority_class = training_set[training_set['Sentiment'] == 0]
    majority_class = training_set[training_set['Sentiment'] == 1]
    # sample without replacement, matching minority class size
    majority_class_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class))
    # concat the downsamppled majority class with the minority class
    downsampled_training_set = pd.concat([minority_class, majority_class_downsampled])
    df = pd.concat([downsampled_training_set, test_set])
    return downsampled_training_set, df


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
    word_count_by_y = get_word_count_by_y()
    # X train is binary bag of words. data has 0 or 1
    X_train_coo = X_train.tocoo()
    feature_counts_positive = np.zeros(X_train.shape[1])
    feature_counts_negative = np.zeros(X_train.shape[1])

    # i is review number (row in data set)
    # j is vocabulary index
    # v is presence of word j in row i (0/1 as it is binary bag of words)
    for i, j, v in zip(X_train_coo.row, X_train_coo.col, X_train_coo.data):
        # if negative sentiment
        if y_train[i] == 0:
            feature_counts_negative[j] += v
        else:
            #if positive sentiment
            feature_counts_positive[j] += v
    # for each word and label get probability =  ((count x = X and y = Y)+1) / (count(number of words when y = Y) + # of words in Vocabulary)
    # +1 smoothing
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
    predictions = np.zeros(X.shape[0])
    # use log as per instructions
    log_probability_negative = np.log(probability_negative)
    log_probability_positive = np.log(probability_positive)
    log_likelihoods_negative = np.log(likelihoods_negative)
    log_likelihoods_positive = np.log(likelihoods_positive)
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
    posterior_probability_positive = np.exp(log_posterior_probability_positive)
    posterior_probability_negative = np.exp(log_posterior_probability_negative)
    return predictions, posterior_probability_positive, posterior_probability_negative


def print_metrics(y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm)
    plot_ROC_curve(y_pred, y_test)
    TP, FP = cm[0]
    FN, TN = cm[1]

    # Calculate metrics
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    npv = TN / (TN + FN)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    f_score = 2 * (precision * sensitivity) / (precision + sensitivity)

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
    class_names = ['Negative', 'Positive']
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def plot_ROC_curve(y_pred, y_test):
    # Assuming you have already trained your model and obtained predicted probabilities and true labels
    # predicted_probabilities: Predicted probabilities of the positive class
    # true_labels: True labels (0 or 1)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Calculate area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
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
    df = pd.read_csv("DisneylandReviews.csv")
    clean_data()

    # Step 2 training classifier
    print("Training classifier...")
    # step 2.2 split into training and test sets
    training_set, test_set = get_training_and_test_sets()
    # step 2.2 (model split for presentation)
    training_set, df = undersample_training_set()
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
