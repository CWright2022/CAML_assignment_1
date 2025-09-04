# This file trains and tests a KNN classifier to detect spam vs. ham SMS messages

import os
import math
from random import shuffle

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


def calculate_sms_idf(vocabulary: list[str], sms: str, sms_list: list[str]) -> list[float] | None:
    """
    Calculate the inverse document frequency for a given SMS message
    For each token in the vocabulary: log( (total # of SMS messages) / (# of messages this word appears in) ) for each word in this SMS
    If a word is in the vocabulary but not in this SMS, it gets a 0

    Args:
        vocabulary (list[str]): set of all words present in the samples
        sms (str): the given SMS message
        sms_list (list[str]): list of all SMS messages
    Returns:
        list[float]: inverse document frequency calculations given this message for each word in our vocabulary and this set of SMS messages
        Will return None if the given SMS has no tokens. For example, with certain cleaning methods an SMS will have no tokens
    """
    inverse_document_frequency_list = []
    tokens = tokenize_sms(sms)
    token_length = len(tokens)
    if token_length == 0:
        return None
    


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
    vocabulary = list(set(ham_tokens + spam_tokens))

    # Calculate term frequencies and inverse document frequencies for each SMS in the training set
    sms_term_frequencies = {}
    sms_inverse_document_frequencies = {}
    for sms in sms_samples.keys():
        sms_tf = calculate_sms_tf(vocabulary, sms)
        if sms_tf is None:
            continue
        sms_term_frequencies[sms] = sms_tf

        sms_idf = calculate_sms_idf(vocabulary, sms, list(sms_samples.keys()))






if __name__ == '__main__':
    main()