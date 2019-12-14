# Naive Bayes On The Wine Dataset
import random
from math import sqrt
from math import exp
from math import pi
from sklearn.datasets import load_wine
import numpy as np


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    random.shuffle(dataset)
    len_data = []
    fold_size = int(len(dataset) / n_folds)
    for i in range(1, n_folds):
        len_data.append(i * fold_size)
    return np.split(dataset, len_data)


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def combine(folds, i):
    ans = []
    for z in range(len(folds)):
        if z != i:
            ans.extend(folds[i])
    return ans


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for i in range(len(folds)):
        train_set = combine(folds, i)
        test_set = list()
        for row in folds[i]:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in folds[i]]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del (summaries[-1])
    return summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# Calculate the Gaussian probability distribution function for x
def Gaussian_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= Gaussian_probability(row[i], mean, stdev)
    return probabilities


# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# Naive Bayes Algorithm
def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return predictions


def load_data():
    dataset = []
    [feature, label] = load_wine(return_X_y=True)
    for i in range(len(feature)):
        temp = []
        temp.extend(feature[i])
        temp.append(int(label[i]))
        dataset.append(temp)
    return dataset


dataset = load_data()
# evaluate algorithm
n_folds = 6
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

