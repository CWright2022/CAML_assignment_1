# This file trains and tests a Naive Bayes Classifier to detect spam vs. ham SMS messages

from random import shuffle
import os

# Color codes
COLOR_DEFAULT = '\x1b[39m'
COLOR_GREEN = '\x1b[32m'
COLOR_RED = '\x1b[31m'

# File name containing the data from which to train and test
DATA_FILENAME = 'sms_data.txt'

# Naive Bayes alpha value
ALPHA = 1

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


def calculate_probabilities(vocabulary: list[str], sms_tokens: list[str]) -> dict[str, float]:
    """
    Calculates the probabilities for each word given it is that type of message (spam/ham)

    Args:
        vocabulary (list[str]): list of all words in our vocabulary (all words appearing in the training set)
        sms_list (list[str]): list of SMS messages from either the ham or spam set
    Returns:
        dict[str, float]: mapping of each word to the probability
    """
    # all counts get initialized to ALPHA
    token_counts = {token: ALPHA for token in vocabulary}
    for token in sms_tokens:
        token_counts[token] += 1

    total_count = sum(token_counts.values())
    probabilities = {token: count/total_count for token, count in token_counts.items()}
    return probabilities


def calculate_posterior_probability(sms: str, prior_probability: float, probabilities: dict[str, float]) -> float:
    """
    Calculates the posterior probability (prior_probability * P(w | spam/ham) for each word in the sms)

    Args:
        sms (str): the SMS message to check
        prior_probability (float): prior probability value to use in the calculation
        probabilities (dict[str, float]): contains necessary probabilities for each token
    Returns:
        float: the posterior probability
    """
    tokens = tokenize_sms(sms)
    posterior_probability = prior_probability
    for token in tokens:
        # default of 1 ensures if we've never seen this token before, it will be ignored
        token_probability = probabilities.get(token, 1)
        posterior_probability *= token_probability
    return posterior_probability


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

    # calculate prior probabilities
    prior_ham_probability = len(ham_list) / (len(ham_list) + len(spam_list))
    prior_spam_probability = len(spam_list) / (len(ham_list) + len(spam_list))

    # create train/test sets using stratified split
    ham_list_train = ham_list[:int(TRAINING_SPLIT*len(ham_list))]
    ham_list_test = ham_list[int(TRAINING_SPLIT*len(ham_list)):]
    spam_list_train = spam_list[:int(TRAINING_SPLIT*len(spam_list))]
    spam_list_test = spam_list[int(TRAINING_SPLIT*len(spam_list)):]

    ham_tokens = get_tokens(ham_list_train)
    spam_tokens = get_tokens(spam_list_train)
    vocabulary = list(set(ham_tokens + spam_tokens))

    # probability for each word given the sms is ham
    ham_probabilities = calculate_probabilities(vocabulary, ham_tokens)
    
    # probability for each word given the sms is spam
    spam_probabilities = calculate_probabilities(vocabulary, spam_tokens)


    # Testing
    ground_truth = {}
    for sms in ham_list_test:
        ground_truth[sms] = 0
    for sms in spam_list_test:
        ground_truth[sms] = 1

    results = {}
    for sms in ground_truth.keys():
        ham_posterior_probability = calculate_posterior_probability(sms, prior_ham_probability, ham_probabilities)
        spam_posterior_probability = calculate_posterior_probability(sms, prior_spam_probability, spam_probabilities)
        results[sms] = 0 if ham_posterior_probability > spam_posterior_probability else 1
    
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
