from __future__ import unicode_literals, print_function
import glog


class Instance(object):
    # Split point for meta information and features.
    META_FEATURE_DIVIDE = 4

    def __init__(self):
        # Unique id among the data set, e.g., training set.
        self.id = None
        # Document id.
        self.doc_id = None
        # Sentence id, we don't do cross-sentence relation extraction.
        self.sentence_id = None
        # Initialize as NONE relation, which should be removed if valid
        # relation is added.
        # self.relations = {spec.RelationType.NONE}
        # We should initialize it as empty list to indicate it is NONE
        # relation.
        self.positive_relations = set()
        # Unique id for the entity pair, normally formed by entity
        # normalization ids.
        self.entity_pair_id = None
        # A list of features.
        self.features = []
        # A list of features that are not modified all time.
        self.original_features = []
        # A tagged sentence used for displaying the entity pair.
        self.tagged_sentence = None

    def __repr__(self):
        if len(self.positive_relations) > 0:
            relations_repr = '|'.join(sorted(self.positive_relations))
        else:
            relations_repr = str(None)
        return '{}\t{}'.format(relations_repr, self.tagged_sentence)

    def is_none_relation(self):
        return len(self.positive_relations) == 0

    def relation_string(self):
        if len(self.positive_relations) > 0:
            return '|'.join(sorted(self.positive_relations))
        else:
            return 'NONE'

    def serialize_to_line(self):
        meta = [self.relation_string(), self.doc_id,
                str(self.sentence_id), self.entity_pair_id]
        instance_tuple = meta + self.original_features
        return ' '.join(instance_tuple) + '\t' + self.tagged_sentence.replace('\n', ' ')

    @staticmethod
    def parse_from_line(line):
        line = line.strip()
        try:
            info, tagged = line.split('\t')
            bits = info.split(' ')
            meta = bits[:Instance.META_FEATURE_DIVIDE]

            instance = Instance()

            # Load meta information
            relations = meta[0].split('|')
            for rel in relations:
                if rel == 'NONE':
                    continue
                instance.positive_relations.add(rel)
        except:
            glog.warning('Parsing instance from line error: {}'.format(line))
            return

        instance.doc_id, instance.sentence_id, instance.entity_pair_id = meta[1:]
        instance.tagged_sentence = tagged
        # Load instance features, usually will be filtered by
        # other conditions.
        instance.features = bits[Instance.META_FEATURE_DIVIDE:]
        instance.original_features = bits[Instance.META_FEATURE_DIVIDE:]
        return instance


