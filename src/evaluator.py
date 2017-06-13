from __future__ import print_function, division
import os
import math
import codecs
import pickle
from itertools import cycle
from collections import namedtuple
from collections import defaultdict
import glog
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


def dictionarize_examples(examples):
    for example in examples:
        dict_example = {}
        for f in example:
            if f in dict_example:
                dict_example[f] += 1
            else:
                dict_example[f] = 1
        return dict_example


def multi_instance_predict(learners, examples):
    fold_prediction_probs = {}
    for relation in learners[0].model.classes_:
        fold_prediction_probs[relation] = [0] * len(examples)

    for l in learners:
        # predictions.append(l.predict(examples))
        probs = l.predict_prob(examples)
        for i, label_probs in enumerate(probs):
            for label, prob in label_probs:
                fold_prediction_probs[label][i] += prob / len(learners)

    predictions = []
    prediction_probs = []

    for i in range(len(examples)):
        single_prob = []
        for relation, relation_probs in fold_prediction_probs.items():
            single_prob.append((relation, relation_probs[i]))

        best_predict = max(single_prob, key=lambda a: a[1])
        predictions.append(best_predict[0])
        prediction_probs.append(best_predict[1])

    # Lists of instances, predictions and their probabilities.
    return predictions, prediction_probs


def generate_prec_recall_points(clf, test_examples, test_labels, pk_file):
    # Generate precision-recall points and store in a pickle file.

    precision = dict()
    recall = dict()
    average_precision = dict()
    thresholds = dict()

    n_classes = len(clf.model.classes_)
    y_test = label_binarize(test_labels, clf.model.classes_)

    y_score = clf.predict_raw_prob(test_examples)
    # It only output 1 column of positive probability.
    y_score = y_score[:, 1:]

    for i in range(n_classes - 1):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(
            y_test[:, i],
            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i],
                                                       y_score[:, i])
    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], thresholds['micro'] = \
        precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                         average="micro")

    if pk_file is not None:
        with open(pk_file, 'wb') as f:
            pickle.dump((precision, recall, average_precision, thresholds), f)


