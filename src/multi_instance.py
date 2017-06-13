from __future__ import print_function, division
import sys
import math
import pickle
from random import shuffle
from collections import defaultdict

import glog
from iterator import MultiInstanceIterator, MultiInstanceFoldIterator
from evaluator import multi_instance_predict
from learner import LogisticRegressionLearner as LogReg
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


class MultiInstanceLearner(object):
    def __init__(self, name, relation_name, iter_train, fold=4):
        assert fold > 1

        self.name = name
        # The positive relation label.
        self.relation_name = relation_name
        # A base multi-class learner for all relations and N binary
        # learner for each relation.
        self.multi_class_learner = None
        self.binary_learner = None

        # User should define the iterator for training data.
        self.multi_iter_train = iter_train
        self.fold = fold
        self.fold_iter = MultiInstanceFoldIterator('train', iter_train, fold)
        self.fold_iter.load_fold()
        self.e_step_result = None

    def create_learner(self):
        self.binary_learner = LogReg(self.name)
        self.multi_class_learner = []
        for i in range(self.fold):
            # Avoid train and test on the same data in the EM process.
            self.multi_class_learner.append(LogReg('MULTICLASS_{}'.format(i)))

    def init_multiclass_classifier(self):
        msg = 'Initializing {} multiclass classifiers...'.format(self.fold)
        glog.info(msg)

        for i in range(self.fold):
            fold_examples = []
            fold_labels = []
            for instances, labels in \
                    self.fold_iter.iter_other_fold_as_group_relation_pair(i):
                for instance in instances:
                    fold_examples.append(instance.features)
                    fold_labels.append(labels[0])

            # 20 iterations.
            self.multi_class_learner[i].learn(fold_examples, fold_labels, 50)

        msg = 'Initialized {} multiclass classifiers'.format(self.fold)
        glog.info(msg)

    # MIML-RE version
    def init_binary_classifier(self):
        msg = 'Initializing binary classifier'
        glog.info(msg)

        # We initialize the binary classifiers such that a relation
        # has a high weight for itself. Since other relation appearing
        # including the NONE relation doesn't neccessary mean the
        # entity pair has no current relation.
        examples = [[self.relation_name], ['NONE']]

        # 20 iterations.
        self.binary_learner.learn(examples, [self.relation_name, 'NONE'], 20)
        # probs = learner.predict_prob(examples)
        # glog.info('{} {}'.format(examples, probs))


    # Riedel version
    # def init_binary_classifier(self):
    #     examples, labels = [], []
    #     for instance, label in self.multi_iter_train.iter_as_instance_relation_pair():
    #         examples.append(instance.features)
    #         if label is not None:
    #             labels.append(label.name)
    #         else:
    #             labels.append('NONE')
    #
    #     for relation, learner in self.binary_learners.items():
    #         msg = 'Initializing binary classifier {}...'.format(relation.name)
    #         glog.info(msg)
    #
    #         # 20 iterations.
    #         learner.learn(examples, labels, 20)
    #         # probs = learner.predict_prob(examples)
    #         # glog.info('{} {}'.format(examples, probs))
    #
    #     msg = 'Initialized {} binary classifiers'.format(
    #         len(self.binary_learners))
    #     glog.info(msg)


    def e_step(self, fold):
        glog.info('Fold {}, running E step...'.format(fold))
        e_step_result = []
        z_changed = defaultdict(int)
        high_conf = 0

        for instances, group_relations in \
                self.fold_iter.iter_fold_as_group_relation_pair(fold):
            examples = [instance.features for instance in instances]

            # Randomize the instances before doing the E step, since the z
            # labels depend on each other. We don't want to introduce
            # some bias due to the specific iteration order of the instances.
            shuffle(examples)

            z_values = self.multi_class_learner[fold].predict(examples)
            z_probs = self.multi_class_learner[fold].predict_prob(examples)
            assert len(z_values) == len(z_probs)

            for v, p in zip(z_values, z_probs):
                for vv, pp in p:
                    if vv == v:
                        if pp > 0.9:
                            high_conf += 1
                        break

            for i, example in enumerate(examples):
                z_joint_probs = []
                z_label_probs = z_probs[i]

                for mention_label, z_prob in z_label_probs:
                    curr = z_values[:i] + [mention_label] + z_values[i + 1:]
                    z_joint_prob = math.log(z_prob)

                    # This is MIML-RE features.
                    # Get features for all binary classifiers, following
                    # the at-least-one assumption.
                    features = set(curr)
                    if len(features) > 1:
                        # At-least-one assumption. If there is a positive
                        # z-label, then remove the NONE relation prediction.
                        features.discard('NONE')
                    features = list(features)

                    # This is Riedel features.
                    # features = []
                    # for j, curr_label in enumerate(curr):
                    #     if curr_label == 'NONE':
                    #         continue
                    #     # Only use features in active instances.
                    #     features += examples[j]

                    # Adding probability for all the positive and negative
                    # relations for joint inference.

                    assert len(group_relations) == 1
                    if len(group_relations) > 0:
                        gold = group_relations[0]
                    else:
                        gold = 'NONE'

                    y_label_probs = self.binary_learner.predict_one_prob(features)
                    y_matched = False
                    for y, y_prob in y_label_probs:
                        if y == gold:
                            if y_prob > 0:
                                z_joint_prob += math.log(y_prob)
                            else:
                                z_joint_prob += float('-inf')
                            y_matched = True
                            break
                    assert y_matched

                    z_joint_probs.append((mention_label, z_joint_prob))

                # Get the best z label based on joint inference.
                best_z_prob = max(z_joint_probs, key=lambda a: a[1])

                if z_values[i] != best_z_prob[0]:
                    change = '{} -> {}'.format(z_values[i], best_z_prob[0])
                    z_changed[change] += 1

                z_values[i] = best_z_prob[0]
            # Add the inferred z labels for the M step.
            e_step_result.append((examples, z_values, group_relations))

        self.e_step_result.append(e_step_result)
        # During E step, some z labels will change due to the joint inference.
        glog.info('Mention label changes: {}'.format(z_changed))
        glog.info('>0.9 instances: {}'.format(high_conf))

    def m_step(self, fold):
        glog.info('Fold {}, running M step...'.format(fold))
        multi_class_data = []
        multi_class_label = []
        binary_class_data = []
        binary_class_label = []

        e_step_result = []
        instance_count = 0
        label_count = 0
        for i in range(self.fold):
            if i == fold:
                continue
            for eres in self.e_step_result[i]:
                e_step_result.append(eres)
                instance_count += len(eres[0])
                label_count += len(eres[1])
                assert instance_count == label_count, (
                    instance_count, label_count)

        for examples, z_values, group_relations in e_step_result:
            multi_class_data += examples
            multi_class_label += z_values

            y_features = list(set(z_values))
            binary_class_data.append(y_features)
            if len(group_relations) == 0:
                binary_class_label.append('NONE')
            else:
                assert len(group_relations) == 1
                binary_class_label.append(group_relations[0])

        # Train multi-class classifier.
        self.multi_class_learner[fold].learn(multi_class_data,
                                             multi_class_label, 50)

        # Train binary classifiers.
        self.binary_learner.learn(binary_class_data,
                                  binary_class_label, 10)


    def learn(self, epoch=10):
        for i in range(epoch):
            glog.info('Epoch {}'.format(i))
            self.e_step_result = []
            for f in range(self.fold):
                self.e_step(f)
            for f in range(self.fold):
                self.m_step(f)


if __name__ == '__main__':
    glog.setLevel(glog.DEBUG)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    eval_file = sys.argv[3]
    # Used for training the binary classifier at entity pair level.
    relation_name = sys.argv[4]


    # Iterator for training data.
    iter_train = MultiInstanceIterator('train')
    iter_train.base_iter.set_min_feature_freq(2)

    iter_train.base_iter.set_feature_set('miml')
    iter_train.base_iter.log_setting()

    iter_train.base_iter.load_instance(train_file)
    iter_train.load_as_training_data(shuffle_entity_pair=True)

    # Iterator for test set. We don't set group limit
    # since this may be some human annotated corpus, so
    # that we should evaluate all instances.
    iter_test = MultiInstanceIterator('test')
    iter_test.base_iter.set_feature_set('miml')

    # Load data.
    iter_test.base_iter.load_instance(test_file)
    iter_test.load_as_test_data()

    # Create the learner.
    milearner = MultiInstanceLearner('train', relation_name, iter_train, fold=4)

    # Init.
    milearner.create_learner()
    milearner.init_multiclass_classifier()
    milearner.init_binary_classifier()

    # Training for 5 epochs.
    milearner.learn(5)

    examples = []
    gold = []
    for instance, gold_label in iter_test.iter_as_instance_relation_pair():
        examples.append(instance.features)
        gold.append(gold_label)

    predictions, probs = multi_instance_predict(
        milearner.multi_class_learner, examples)

    new_probs = []
    for prec, prob in zip(predictions, probs):
        if prec == 'NONE':
            new_probs.append(1-prob)
        else:
            new_probs.append(prob)

    # This will be a column vector where 1=REL and 0=NONE
    # new_prob will be a column vvector where element are prob of REL.
    gold_bin = label_binarize(gold, milearner.multi_class_learner[0].model.classes_)
    precision, recall, thresholds, average_precision = {}, {}, {}, {}
    precision[0], recall[0], thresholds[0] = precision_recall_curve(gold_bin, new_probs)
    average_precision[0] = average_precision_score(gold_bin, new_probs)
    with open(eval_file, 'wb') as f:
        pickle.dump((precision, recall, average_precision, thresholds), f)
