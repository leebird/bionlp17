from __future__ import unicode_literals, print_function
import sys
import codecs
from collections import defaultdict
import pprint
import re


if __name__ == '__main__':
    train_file = sys.argv[1]
    filtered_file = sys.argv[2]
    top_num = int(sys.argv[3])
    target_relation = sys.argv[4]
    top_trigger = sys.argv[5]

    trigger_file = 'data/triggers/{}.txt'.format(target_relation)

    trigger_set = set()
    if top_trigger > 0:
        with codecs.open(trigger_file, encoding='utf8') as f:
            for line in f:
                line = line.strip()
                trigger, count = line.split('\t')
                trigger_set.add(trigger)
                if len(trigger_set) >= top_trigger:
                    break

    filtered = 0
    instances = []
    split_pattern = re.compile('<-|->')
    pattern_count = defaultdict(int)
    with codecs.open(train_file, 'r', 'utf8') as f:
        for line in f:
            line = line.strip()
            try:
                tokens, tagged = line.split('\t')
            except:
                continue
            instances.append(line)
            tokens = tokens.split(' ')
            relation = tokens[0]
            if relation != 'NONE':
                pattern = [p for p in tokens if p.startswith('lex_dep_path:')]
                for p in pattern:
                    parts = re.split(split_pattern, p)
                    for part in parts:
                        if part.startswith('w:'):
                            word = part[2:]
                            if word in trigger_set:
                                pattern_count[p] += 1

    sorted_patterns = sorted([(p, c) for p, c in pattern_count.items()],
                             key=lambda a: a[1],
                             reverse=True)

    # pprint.pprint(sorted_patterns[:top_num])
    top_patterns = set([p for p, c in sorted_patterns[:top_num]])

    with codecs.open(filtered_file, 'w', 'utf8') as f:
        for line in instances:
            tokens, tagged = line.split('\t')
            tokens = tokens.split(' ')
            relation = tokens[0]
            if relation == 'NONE':
                pattern = [p for p in tokens if p.startswith('lex_dep_path:')]
                if len(pattern) > 0:
                    remove = False
                    for p in pattern:
                        if p in top_patterns:
                            remove = True
                            break

                    if remove:
                        filtered += 1
                        continue
            f.write(line+'\n')

    print('Using {} High-confidence patterns'.format(len(top_patterns)))
    print('Filtered {} negative instaces'.format(filtered))
