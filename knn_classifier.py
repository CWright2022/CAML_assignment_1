# This file trains and tests a KNN classifier to detect spam vs. ham SMS messages
# Requires Python 3.10+

import os
import math
from random import shuffle
import numpy as np

# Color codes
COLOR_DEFAULT = '\x1b[39m'
COLOR_GREEN = '\x1b[32m'
COLOR_RED = '\x1b[31m'

# File name containing the data from which to train and test
DATA_FILENAME = 'sms_data.txt'

# K value to use for the KNN
K = 1

# proportion of messages that will be used for training. The remaining will be used for testing
TRAINING_SPLIT = 0.8

def tokenize_sms(sms: str) -> list[str]:
    """
    Takes a sms message and splits it into clean tokens

    Args:
        sms (str): SMS message to be tokenized
    Returns:
        list[str]: list of resultant tokens
    """
    # remove all non-letter characters from the sms
    sms = ''.join(let for let in sms if let.isalnum() or let == ' ')
    sms = sms.lower()
    return sms.split()


def load_data(data_filename: str) -> tuple[list[str], list[str]]:
    """
    Returns list of spam messages and list of ham messages

    Args:
        data_filename (str): data filename from which to read
    Returns:
        list[str]: list of ham messages
        list[str]: list of spam messages
    """
    ham_list = []
    spam_list = []
    if not os.path.exists(data_filename):
        raise FileNotFoundError(f'The file "{data_filename}" could not be found.')
    with open(data_filename, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if fields[0] == 'ham':
                ham_list.append(fields[1])
            elif fields[0] == 'spam':
                spam_list.append(fields[1])
            else:
                raise ValueError(f'Unrecognized classification in data file: {fields[0]}')
    return ham_list, spam_list


def get_tokens(sms_list: list[str]) -> list[str]:
    """
    Extract all tokens from a list of sms. Returns a flat list

    Args:
        sms_list (list[str]): list of sms messages
    Returns:
        list[str]: flat list of all the tokens, including repeats
    """
    tokens_list = []
    for sms in sms_list:
        sms_tokens = tokenize_sms(sms)
        tokens_list.extend(sms_tokens)
    return tokens_list


def calculate_sms_tf(vocabulary: list[str], sms: str) -> list[float] | None:
    """
    Calculates the term frequency for a SMS message
    For each unique token in this message, calculates what proportion of the tokens in the message are that token
    If a token appears in the vocabulary but not in this message, it gets a 0

    Args:
        vocabulary (list[str]): set of all words present in the samples
        sms (str): the given SMS message
    Returns:
        list[float]: term frequency calculations given this message for each word in our vocabulary
        Will return None if the given SMS has no tokens. For example, with certain cleaning methods an SMS will have no tokens
    """
    term_frequency_list = []
    tokens = tokenize_sms(sms)
    token_length = len(tokens)
    if token_length == 0:
        return None
    token_counts = dict()
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    for token in vocabulary:
        term_frequency_list.append(token_counts.get(token, 0) / token_length)
    return term_frequency_list


def calculate_sms_idf(vocabulary: list[str], sms: str, tokenized_sms_list: list[list[str]], testing_sample=False) -> list[float] | None:
    """
    Calculate the inverse document frequency for a given SMS message
    For each token in the vocabulary: log( (total # of SMS messages) / (# of messages this word appears in) ) for each word in this SMS
    If a word is in the vocabulary but not in this SMS, it gets a 0

    Args:
        vocabulary (list[str]): set of all words present in the samples
        sms (str): the given SMS message
        sms_list (list[str]): list of all SMS messages
        testing_sample (bool=False): if this sample is not in the training set, must add one to the denominator
    Returns:
        list[float]: inverse document frequency calculations given this message for each word in our vocabulary and this set of SMS messages
        Will return None if the given SMS has no tokens. For example, with certain cleaning methods an SMS will have no tokens
    """
    inverse_document_frequency_list = []
    tokenized_sms = tokenize_sms(sms)
    token_length = len(tokenized_sms)
    if token_length == 0:
        return None
    document_total = len(tokenized_sms_list)
    # Calculate idf for this sms
    for token in vocabulary:
        if token in tokenized_sms:
            token_appearance_count = 1 if testing_sample else 0
            for tokens in tokenized_sms_list:
                if token in tokens:
                    token_appearance_count += 1
            inverse_document_frequency_list.append(math.log10(document_total / token_appearance_count))
        else:
            inverse_document_frequency_list.append(0)
    return inverse_document_frequency_list


def calculate_n_dimensional_distance(p1: list[float], p2: list[float]) -> float:
    """
    Calculates the distance between two points in n-dimensional space
    Uses numpy in order to be more efficient

    Args:
        p1 (list[float]): the first point
        p2 (list[float]): the second point
    Returns:
        float: The distance between the points
    """
    if len(p1) != len(p2):
        raise ValueError('The two provided points do not have the same dimensionality')
    n1 = np.array(p1)
    n2 = np.array(p2)
    result = np.sqrt(np.sum((n1 - n2)**2))
    return result


def main():
    # Training
    try:
        ham_list, spam_list = load_data(DATA_FILENAME)
    except FileNotFoundError as e:
        print(e)
        exit()

    # shuffle so training/testing split is random each time
    shuffle(ham_list)
    shuffle(spam_list)

    # create train/test sets using stratified split
    ham_list_train = ham_list[:int(TRAINING_SPLIT*len(ham_list))]
    ham_list_test = ham_list[int(TRAINING_SPLIT*len(ham_list)):]
    spam_list_train = spam_list[:int(TRAINING_SPLIT*len(spam_list))]
    spam_list_test = spam_list[int(TRAINING_SPLIT*len(spam_list)):]

    sms_samples = {}
    for sms in ham_list_train:
        sms_samples[sms] = 0
    for sms in spam_list_train:
        sms_samples[sms] = 1

    ham_tokens = get_tokens(ham_list_train)
    spam_tokens = get_tokens(spam_list_train)

    # Convert to list so ordering stays consistent
    # The order of the vocabulary is the order the tokens must show up in the frequency dictionaries
    vocabulary = list(set(ham_tokens + spam_tokens))

    # Pre-calculate this for the idf function
    tokenized_sms_list = []
    for sms in list(sms_samples.keys()):
        tokenized_sms_list.append(tokenize_sms(sms))

    # Calculate term frequencies and inverse document frequencies for each SMS in the training set
    # Also calculate the TF-IDF by multiplying both values together
    sms_term_frequencies = {}
    sms_inverse_document_frequencies = {}
    sms_tf_idf = {}
    for sms in list(sms_samples.keys()):
        sms_tf = calculate_sms_tf(vocabulary, sms)
        if sms_tf is None:
            continue
        sms_term_frequencies[sms] = sms_tf

        sms_idf = calculate_sms_idf(vocabulary, sms, tokenized_sms_list)
        if sms_idf is None:
            continue
        sms_inverse_document_frequencies[sms] = sms_idf

        sms_tf_idf[sms] = [tf * idf for tf, idf in zip(sms_tf, sms_idf)]
    
    # Testing
    ground_truth = {}
    for sms in ham_list_test:
        ground_truth[sms] = 0
    for sms in spam_list_test:
        ground_truth[sms] = 1
    
    results = {}
    for sms in list(ground_truth.keys()):
        tf = calculate_sms_tf(vocabulary, sms)
        idf = calculate_sms_idf(vocabulary, sms, tokenized_sms_list, True)
        tf_idf = list(tf_val * idf_val for tf_val, idf_val in zip(tf, idf))

        # Calculate the distance between this vector and the vector for every point in the training set
        distances = {}
        print('doing calc')
        for training_sms, training_tf_idf in sms_tf_idf.items():
            distances[training_sms] = calculate_n_dimensional_distance(training_tf_idf, tf_idf)
        print('finish calc')
        # Sort dictionary by value and select the K lowest values
        # If spam/ham has the majority in the K lowest values, then that is our prediction
        distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        ham_count = 0
        spam_count = 0
        for sms in list(distances.keys())[:K]:
            if sms_samples[sms] == 0:
                ham_count += 1
            else:
                spam_count += 1
        results[sms] = 0 if ham_count > spam_count else 1
        print(f'{sms[:150]}...')
        print('ham' if ham_count > spam_count else 'spam')
        print()
        
    # Results and performance metrics
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for sms, prediction in results.items():
        print(f'{sms[:150]:<160}', end='')
        if prediction:
            print(f'{COLOR_RED}SPAM{COLOR_DEFAULT}\t', end='')
            if ground_truth[sms] == prediction:  # True positive
                print(f'{COLOR_GREEN}TP{COLOR_DEFAULT}')
                tp += 1
            else:  # False positive
                print(f'{COLOR_RED}FP{COLOR_DEFAULT}')
                fp += 1
        else:
            print(f'{COLOR_GREEN}HAM{COLOR_DEFAULT}\t', end='')
            if ground_truth[sms] == prediction:  # True negative
                print(f'{COLOR_GREEN}TN{COLOR_DEFAULT}')
                tn += 1
            else:  # False negative
                print(f'{COLOR_RED}FN{COLOR_DEFAULT}')
                fn += 1

    print()
    print('Totals:')
    print(f'True positives: {tp}')
    print(f'False positives: {fp}')
    print(f'True negatives: {tn}')
    print(f'False negatives: {fn}')

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)

    print()
    print('Metrics:')
    print(f'Accuracy: {accuracy*100:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall: {recall*100:.2f}%')
    print(f'F1-score: {f1_score:.2f}')

            




if __name__ == '__main__':
    main()