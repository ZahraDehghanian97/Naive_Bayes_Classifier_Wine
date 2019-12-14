# Naive Bayes On The Wine Dataset
import random
from math import sqrt
from math import exp
from math import pi
from sklearn.datasets import load_wine
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc


# Evaluate an algorithm
def one_vs_all(actual, predicted, num_class):
    y_real = []
    y_predict = []
    for i in range(num_class):
        temp_r = []
        temp_p = []
        positive = i
        for j in range (len(actual)):
            if actual[j] == positive :
                temp_r.append(1)
            else :
                temp_r.append(0)
            if predicted[j]== positive:
                temp_p.append(1)
            else :
                temp_p.append(0)
        y_real.append(temp_r)
        y_predict.append(temp_p)
    return y_real,y_predict


def evaluate_algorithm(dataset, algorithm,num_class, *args):
    random.shuffle(dataset)
    train_set, fold = np.split(dataset, [int(2 * len(dataset) / 3)])
    test_set = list()
    for row in fold:
        row_copy = list(row)
        test_set.append(row_copy)
        row_copy[-1] = None
    predicted = algorithm(train_set, test_set, *args)
    actual = [row[-1] for row in fold]
    y_real , y_predict = one_vs_all(actual,predicted,num_class)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(np.array(y_real)[:, i],np.array(y_predict)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(num_class):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
    plt.show()


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
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
def calculate_probability(x, mean, stdev):
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
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
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
evaluate_algorithm(dataset, naive_bayes,3)
