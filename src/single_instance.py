from __future__ import unicode_literals, print_function, division

import sys
from iterator import MultiInstanceIterator
from learner import LogisticRegressionLearner
from evaluator import generate_prec_recall_points
import glog


class Learner(object):
    def __init__(self, iter_train):
        self.iter_train = iter_train
        self.learner = LogisticRegressionLearner('train', False)

    def learn(self):
        examples = []
        labels = []
        for instance, relation in \
                self.iter_train.iter_as_instance_relation_pair():
            examples.append(instance.features)
            labels.append(relation)
        self.learner.learn(examples, labels)


def learn_and_test(iter_train, iter_test, pk_file):
    iter_train.load_as_training_data()

    # Learning.
    learner = Learner(iter_train)
    learner.learn()

    # Instance-level evaluation.
    features = []
    labels = []
    for instance, relation in iter_test.iter_as_instance_relation_pair():
        labels.append(relation)
        features.append(instance.features)
    generate_prec_recall_points(learner.learner, features, labels, pk_file)


if __name__ == '__main__':
    glog.setLevel(glog.DEBUG)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    pk_file = sys.argv[3]
    filtering = sys.argv[4]

    iter_test = MultiInstanceIterator('test')
    iter_test.base_iter.load_instance(test_file)
    iter_test.load_as_test_data()

    # Iterator for training data.
    iter_train = MultiInstanceIterator('train')

    # Remove features appearing only once.
    iter_train.base_iter.set_min_feature_freq(2)
    iter_train.base_iter.load_instance(train_file)

    if filtering == 'min_pos_dep_path_freq':
        for min_pos_dep_path_freq in [2, 3, 4, 5, 6, 7, 8, 9]:
            iter_train.base_iter.set_min_dep_freq(min_pos_dep_path_freq)
            iter_train.base_iter.log_setting()
            new_file = pk_file + '{}.pk'.format(min_pos_dep_path_freq)
            learn_and_test(iter_train, iter_test, new_file)
    else:
        iter_train.base_iter.log_setting()
        learn_and_test(iter_train, iter_test, pk_file)

