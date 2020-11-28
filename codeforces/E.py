from collections import defaultdict
from math import log
from math import exp


def get_unique_words_in_message(message):
    unique_words = set()
    for word in message:
        if word not in unique_words:
            unique_words.add(word)
    return unique_words


def get_all_unique_words(messages):
    unique_words = set()
    for message in messages:
        for word in message:
            unique_words.add(word)
    return unique_words


def get_statistics(n, messages, classes):
    word_freq = defaultdict(lambda: 0)
    number_of_messages_in_class = [0] * (n + 1)
    for i in range(len(messages)):
        cur_message = messages[i]
        cur_class = classes[i]
        number_of_messages_in_class[cur_class] += 1
        unique_words_in_cur_message = set()
        for word in cur_message:
            if word not in unique_words_in_cur_message:
                unique_words_in_cur_message.add(word)
                word_freq[word, cur_class] += 1
    return number_of_messages_in_class, word_freq


def count_conditional_probabilities(n, messages, classes, alpha):
    number_of_messages_in_class, word_freq = get_statistics(n, messages, classes)
    prob = defaultdict(lambda: 0)
    unique_words = get_all_unique_words(messages)
    for word in unique_words:
        for cur_class in range(1, n + 1):
            prob[word, cur_class] = (word_freq[word, cur_class] + alpha) / (
                    number_of_messages_in_class[cur_class] + 2 * alpha)
    return prob


def predict_message(prob, n, number_of_messages_in_classes, message, unique_words_in_messages, lambdas):
    number_of_messages = n
    prior_prob = [0.0] * (n + 1)
    for cur_class in range(1, n + 1):
        prior_prob[cur_class] = log(number_of_messages_in_classes[cur_class] / number_of_messages)
    cond_prob = [0.0] * (n + 1)
    unique_words_in_message = get_unique_words_in_message(message)
    for word in unique_words_in_messages:
        for cur_class in range(1, n + 1):
            if word in unique_words_in_message:
                cond_prob[cur_class] += log(prob[word, cur_class])
            else:
                cond_prob[cur_class] += log(1 - prob[word, cur_class])
    total_prob = [0.0] * (n + 1)
    for i in range(1, n + 1):
        total_prob[i] = prior_prob[i] + cond_prob[i] + log(lambdas[i - 1])
        total_prob[i] = exp(total_prob[i])
    total_prob_sum = sum(total_prob) - total_prob[0]
    for i in range(1, n + 1):
        total_prob[i] /= total_prob_sum
    return total_prob[1:]


def bayes_model(messages, classes, lambdas, alpha, messages_test):
    n = len(lambdas)
    number_of_messages_in_classes, word_freq = get_statistics(n, messages, classes)
    prob = count_conditional_probabilities(n, messages, classes, alpha)
    unique_words = get_all_unique_words(messages)
    all_total_prob = []
    for message_test in messages_test:
        all_total_prob.append(predict_message(prob, n, number_of_messages_in_classes, message_test, unique_words, lambdas))
    return all_total_prob


k = int(input())
lambdas = list(map(int, input().split()))
alpha = int(input())
n = int(input())
classes = []
messages = []
for i in range(n):
    line = input().split()
    classes.append(int(line[0]))
    messages.append(line[2:])
m = int(input())
messages_test = []
for i in range(m):
    line = input().split()
    messages_test.append(line[1:])
all_total_prob = bayes_model(messages, classes, lambdas, alpha, messages_test)
for total_prob in all_total_prob:
    print(" ".join(map(str, total_prob)))
