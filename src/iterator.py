from __future__ import unicode_literals, print_function, division
import sys
import glog
import codecs
import random
from collections import defaultdict
from feature_filter import *
from instance import Instance


class InstanceIterator(object):
    # This is a single instance iterator, with negative sampling
    # and feature filtering. To iterate throuth multi-instance
    # entity pairs, use MultiInstanceIterator with this class.
    def __init__(self, name):
        self.name = name

        self.feature_set = 'miml'

        self.feature_set_getters = {
            'miml': get_miml_features,
        }

        # Feature counts used to filter out
        self.positive_feature_freq = defaultdict(int)
        self.feature_freq = defaultdict(int)
        self.min_dep_freq = 0
        self.min_feature_freq = 0

        # See feature_filter.py for more details about the feature getters.
        self.feature_set_getter = self.feature_set_getters[self.feature_set]
        self.instances = []

    def reset_corpus(self):
        self.instances = []
        self.feature_freq = defaultdict(int)
        self.positive_feature_freq = defaultdict(int)

    def set_min_dep_freq(self, min_dep_freq):
        self.min_dep_freq = min_dep_freq

    def set_min_feature_freq(self, min_feature_freq):
        self.min_feature_freq = min_feature_freq

    def set_feature_set(self, feature_set):
        assert feature_set in self.feature_set_getters
        self.feature_set = feature_set
        self.feature_set_getter = self.feature_set_getters[feature_set]

    def log_setting(self):
        msgs = [
            '{}: Min DEP frequency: {}'.format(self.name, self.min_dep_freq),
            '{}: Min feature frequency: {}'.format(self.name, self.min_feature_freq),
        ]
        for msg in msgs:
            glog.info(msg)

    def get_feature(self, features):
        features = self.feature_set_getter(features)
        return features

    def remove_low_freq_feature(self, features):
        # Remove low-freq features.
        if self.min_feature_freq > 0:
            new_features = []
            for f in features:
                if self.feature_freq[f] >= self.min_feature_freq:
                    new_features.append(f)
            features = new_features
        return features
    
    def valid_positive_instance(self, features):
        if len(features) == 0:
            # No features, maybe all are low freq and filtered out.
            return False

        if self.min_dep_freq > 0:
            dep_path = get_dep_path(features)
            if dep_path is not None:
                if self.positive_feature_freq[dep_path] < self.min_dep_freq:
                    return False

        return True

    def load_instance(self, filepath):
        self.reset_corpus()
        # We load the whole data with negative sampling.
        skipped, pos_num, neg_num = 0, 0, 0
        for instance in self.iter_instance_file(filepath):
            if instance.is_none_relation():
                neg_num += 1
            else:
                # TODO: this may be problematic for multiple relations.
                # Maybe we should use separate dicts for different relations.
                # positive_feature_freq is not related to how data will be
                # filtered later.

                pos_num += 1
                for feat in instance.features:
                    self.positive_feature_freq[feat] += 1

            for feat in instance.features:
                # feature_freq is related to how data will be filtered.
                self.feature_freq[feat] += 1

            final_features = self.get_feature(instance.features)
            instance.features = final_features
            # Reset the instance id to its position in the sampled list.
            instance.id = len(self.instances)
            self.instances.append(instance)

        glog.info('{}: loaded {} instances'.format(self.name, len(self.instances)))
        glog.info('{}: loaded {} positives, {} negatives'.format(self.name, pos_num, neg_num))

    def iter_as_training_instance(self):
        # For training instance we need to go through a bunch of
        # checks.
        pos_skipped, neg_skipped = 0, 0
        for instance in self.instances:
            # Remove low-freq features if required. This will modifiy features
            # permanently. Reload data from disk if needed.
            instance.features = self.remove_low_freq_feature(instance.features)
            if not instance.is_none_relation() and \
                    not self.valid_positive_instance(instance.features):
                pos_skipped += 1
                continue
            yield instance

        msg = '{}: skipped {} positive instances'.format(self.name, pos_skipped)
        glog.info(msg)

    def iter_as_test_instance(self):
        # We don't need to do any filtering for the test instances.
        for instance in self.instances:
            yield instance

    def iter_instance_file(self, instance_file_path):
        # Normally should not be used directly. Use iter_training_instance()
        # or iter_test_instance() instead.
        glog.info('{}: loading instances from {}'.format(self.name, instance_file_path))
        count = 0
        with codecs.open(instance_file_path, 'r', 'utf8') as f:
            for line in f:
                line = line.strip()
                instance = Instance.parse_from_line(line)
                if instance is None:
                    continue
                # This id should be unique and used to calculate performance,
                # (precision, recall and F-score).
                instance.id = count
                count += 1
                yield instance
        glog.info('{}: loading instances from {} finished'.format(self.name, instance_file_path))


class MultiInstanceIterator(object):
    def __init__(self, name):
        self.name = name
        self.base_iter = InstanceIterator(name)
        # self.base_iter.log_setting()
        self.pair_group = defaultdict(list)
        self.pair_relation = defaultdict(set)

        # The pairs used to iterate all pairs so that each iteration
        # will have the same order. The order should be generated randomly
        # to avoid any bias.
        self.ordered_pairs = None

        self.relations = set()

    def reset_corpus(self):
        self.pair_group = defaultdict(list)
        self.pair_relation = defaultdict(set)
        self.ordered_pairs = None
        self.relations = set()

    def load_as_training_data(self, shuffle_entity_pair=False):
        self.reset_corpus()
        # Load instance and entity-pair level groups.
        # Let user call this function to avoid multiple samplings,
        # so that data is read from disk only once.
        # self.base_iter.load_sampled_instance(filepath)

        for instance in self.base_iter.iter_as_training_instance():
            iid = instance.entity_pair_id
            self.pair_group[iid].append(instance.id)
            # if len(self.pair_relation[iid]) > 0 and len(instance.positive_relations) == 0:
            #     print(instance.serialize_to_line())
            #     raise RuntimeError
            self.pair_relation[iid] |= instance.positive_relations
            self.relations |= instance.positive_relations

        # Randomly generate pair order for this data set.
        self.ordered_pairs = self.pair_group.keys()

        if shuffle_entity_pair:
            random.shuffle(self.ordered_pairs)

        # msg = '{}: loaded {} training entity pairs'.format(self.name, len(self.pair_group))
        # glog.info(msg)

    def load_as_test_data(self):
        self.reset_corpus()
        # Load instance and entity-pair level groups.f
        # Let user call this function to avoid multiple samplings,
        # so that data is read from disk only once.
        # self.base_iter.load_sampled_instance(filepath)
        for instance in self.base_iter.iter_as_test_instance():
            iid = instance.entity_pair_id
            self.pair_group[iid].append(instance.id)
            self.pair_relation[iid] |= instance.positive_relations
            self.relations |= instance.positive_relations

        # Randomly generate pair order for this data set.
        self.ordered_pairs = self.pair_group.keys()
        # Shouldn't shuffle test data.
        # random.shuffle(self.ordered_pairs)

        msg = '{}: loaded {} test entity pairs'.format(self.name, len(self.pair_group))
        glog.info(msg)

    def iter_as_single_instance(self):
        # Iterate through all the instances.
        pair_count, instance_count = 0, 0

        for pair in self.ordered_pairs:
            positions = self.pair_group[pair]

            pair_count += 1
            instance_count += len(positions)

            for pos in positions:
                instance = self.base_iter.instances[pos]
                yield instance

    def iter_as_instance_relation_pair(self):
        # Iterate through all the instances with multi-relations separated.
        # Note that same instance can be iterated more than once if it
        # has more than 1 relation.
        for instance in self.iter_as_single_instance():
            if len(instance.positive_relations) == 0:
                yield instance, 'NONE'
            for relation in instance.positive_relations:
                yield instance, relation

    def iter_as_group_relation_pair(self):
        # Iterate through all the entity pairs.
        pair_count = 0
        example_count = 0

        for pair in self.ordered_pairs:
            positions = self.pair_group[pair]
            relations = self.pair_relation[pair]

            pair_count += 1
            example_count += len(positions)

            examples = []
            # Randomize the instances for multi-instance learning.
            # Don't shuffle here, let the caller do that.
            # shuffle(positions)
            for pos in positions:
                examples.append(self.base_iter.instances[pos])
            if len(relations) == 0:
                yield examples, ['NONE']
            else:
                # For now there should be only 1 positive relation.
                yield examples, list(relations)

        msg = '[{}] Interate through {} entity pairs, {} instances'.format(
                self.iter_as_group_relation_pair.__name__,
                pair_count, example_count)
        glog.info(msg)


class MultiInstanceFoldIterator(object):
    def __init__(self, name, multi_iter, fold):
        assert fold > 1
        self.multi_iter = multi_iter
        self.fold = fold
        self.all_data = []
        glog.info('Fold: {}'.format(fold))

    def load_fold(self):
        msg = 'Loading {} fold data...'.format(self.fold)
        glog.info(msg)

        # Load all data.
        for instances, labels in self.multi_iter.iter_as_group_relation_pair():
            self.all_data.append((instances, labels))
        msg = 'Loaded {} entity pairs'.format(len(self.all_data))
        glog.info(msg)

    def iter_fold_as_group_relation_pair(self, fold):
        assert fold < self.fold

        # Fold step.
        instance_count = 0
        pair_count = 0
        step = int(len(self.all_data) / self.fold)
        for instances, labels in self.all_data[fold * step:(fold + 1) * step]:
            instance_count += len(instances)
            pair_count += 1
            yield instances, labels
        msg = 'Fold {}: iterate over {} instances, {} pairs'.format(
                fold, instance_count, pair_count)
        glog.info(msg)

    def iter_other_fold_as_group_relation_pair(self, excluded_fold):
        assert excluded_fold < self.fold

        instance_count = 0
        pair_count = 0
        # Fold step.
        step = int(len(self.all_data) / self.fold)

        for i in range(self.fold):
            if i == excluded_fold:
                continue
            for instances, labels in self.all_data[i * step:(i + 1) * step]:
                instance_count += len(instances)
                pair_count += 1
                yield instances, labels

        msg = 'Excluded Fold {}: iterate over {} instances, {} pairs'.format(
                excluded_fold, instance_count, pair_count)
        glog.info(msg)
