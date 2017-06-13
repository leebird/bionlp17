from __future__ import unicode_literals, print_function
import re

_cst_separator = re.compile(r'->|<-')
_dep_separator = re.compile(r'->|<-')


def has_feature(features, feature):
    return feature in features


def get_feature_by_group(features, feature_group):
    return [f for f in features if f.startswith(feature_group)]


def get_feature_by_group_list(features, feature_groups):
    result = []
    for feat in features:
        for group in feature_groups:
            if feat.startswith(group):
                result.append(feat)
                break
    return result


def remove_feature_by_group(features, feature_group):
    return [f for f in features if not f.startswith(feature_group)]


def get_dep_path(features):
    for feat in features:
        if feat.startswith('dep_path:'):
            return feat


def get_dep_path_length(features):
    for feat in features:
        if feat.startswith('dep_path_len:'):
            return int(feat.split(':')[1])


def get_dep_path_length_for_path(dep_path):
    return len(_dep_separator.split(dep_path)) - 1


def get_relation_for_dep_path(dep_path):
    path = dep_path.split('|')[1]
    components = _dep_separator.split(path)
    if '__' in components:
        components.remove('__')
    if '' in components:
        components.remove('')
    return components


def validate_instance(features, cst_path_max_length=0, dep_path_max_length=0):
    if 0 < dep_path_max_length < get_dep_path_length(features):
        return False
    return True


def get_miml_features(features):
    return get_feature_by_group_list(features, [
        'dep_path:',
        'mid:',
        '1l_mid:',
        '1r_mid:',
        '1lr_mid:',
        '2l_mid:',
        '2r_mid:',
        '2lr_mid:',
        'mid_seq_len:',
        'dep_path_len:',
        'ewalk:',
        'vwalk:',
    ])

