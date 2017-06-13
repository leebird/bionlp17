from __future__ import unicode_literals, print_function

import glog
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report


class LogisticRegressionLearner(object):
    def __init__(self, name, warm_start=True):
        self.vocal = DictVectorizer()
        self.model = linear_model.LogisticRegression(warm_start=warm_start,
                                                     solver='sag',
                                                     max_iter=200,
                                                     verbose=0,
                                                     penalty='l2',
                                                     n_jobs=4)

    @staticmethod
    def convert_list_to_dict(example):
        dict_example = {}
        for f in example:
            if f in dict_example:
                dict_example[f] += 1
            else:
                dict_example[f] = 1
        return dict_example

    @staticmethod
    def dictionarize_examples(examples):
        for example in examples:
            yield LogisticRegressionLearner.convert_list_to_dict(example)

    def learn(self, train_examples, train_labels, max_iter=None):
        examples = self.dictionarize_examples(train_examples)
        dataset = self.vocal.fit_transform(examples)
        if max_iter is not None:
            self.model.max_iter = max_iter
        self.model.fit(dataset, train_labels)
        glog.info('Iter: {}'.format(self.model.n_iter_))
        glog.info('Intercept: {}'.format(self.model.intercept_))

    def predict(self, test_examples):
        examples = self.dictionarize_examples(test_examples)
        dataset = self.vocal.transform(examples)
        prediction = self.model.predict(dataset)
        return list(prediction)

    def predict_one(self, test_example):
        examples = self.dictionarize_examples([test_example])
        dataset = self.vocal.transform(examples)
        prediction = self.model.predict(dataset)
        return prediction[0]

    def predict_prob(self, test_examples):
        examples = self.dictionarize_examples(test_examples)
        dataset = self.vocal.transform(examples)
        probs = self.model.predict_proba(dataset)
        example_probs = []
        for prob in probs:
            label_probs = []
            for i, val in enumerate(prob):
                label_probs.append((self.model.classes_[i], val))
            example_probs.append(label_probs)
        return example_probs

    def predict_one_prob(self, test_example):
        examples = self.dictionarize_examples([test_example])
        dataset = self.vocal.transform(examples)
        probs = self.model.predict_proba(dataset)
        label_probs = []
        for i, val in enumerate(probs[0]):
            label_probs.append((self.model.classes_[i], val))
        return label_probs

    def predict_raw_prob(self, test_examples):
        examples = self.dictionarize_examples(test_examples)
        dataset = self.vocal.transform(examples)
        probs = self.model.predict_proba(dataset)
        return probs

    def evaluate(self, test_examples, test_labels):
        predictions = self.predict(test_examples)
        print(classification_report(test_labels, predictions))
