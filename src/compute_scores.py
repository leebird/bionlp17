from __future__ import print_function, division
import pickle
import sys
import os

# Compute precision, recall, F-score and specificity.

folder = 'eval/pr_points'

# Used by Table 5 and Fig. 2. 
expriments = {
    # Results in Table 5.
    'basic': [
        ('baseline', '{}.pk'),
        ('DPFreq', '{}_mindepfreq_5.pk'),
        ('multi-instance', '{}_mi.pk'),
        ('H1', '{}_closest.pk'),
        ('H2', '{}_closest_trigger_50.pk'),
        ('H3', '{}_closest_trigger_50_pattern_100.pk'),
    ],
    # Different number of triggers. (Fig. 2 a-c)
    'h2': [
        ('trigger_10', '{}_closest_trigger_10.pk'),
        ('trigger_20', '{}_closest_trigger_20.pk'),
        ('trigger_30', '{}_closest_trigger_30.pk'),
        ('trigger_40', '{}_closest_trigger_40.pk'),
        ('trigger_50', '{}_closest_trigger_50.pk'),
        ('trigger_60', '{}_closest_trigger_60.pk'),
        ('trigger_70', '{}_closest_trigger_70.pk'),
        ('trigger_80', '{}_closest_trigger_80.pk'),
        ('trigger_90', '{}_closest_trigger_90.pk'),
        ('trigger_100', '{}_closest_trigger_100.pk'),
        # Compare with using 100 patterns.
        ('trigger_10_pattern_100', '{}_closest_trigger_10_pattern_100.pk'),
        ('trigger_20_pattern_100', '{}_closest_trigger_20_pattern_100.pk'),
        ('trigger_30_pattern_100', '{}_closest_trigger_30_pattern_100.pk'),
        ('trigger_40_pattern_100', '{}_closest_trigger_40_pattern_100.pk'),
        ('trigger_50_pattern_100', '{}_closest_trigger_50_pattern_100.pk'),
        ('trigger_60_pattern_100', '{}_closest_trigger_60_pattern_100.pk'),
        ('trigger_70_pattern_100', '{}_closest_trigger_70_pattern_100.pk'),
        ('trigger_80_pattern_100', '{}_closest_trigger_80_pattern_100.pk'),
        ('trigger_90_pattern_100', '{}_closest_trigger_90_pattern_100.pk'),
        ('trigger_100_pattern_100', '{}_closest_trigger_100_pattern_100.pk'),
    ],
    # Different number of patterns. (Fig. 2 d-f)
    'h3': [
        ('trigger_50_pattern_20', '{}_closest_trigger_50_pattern_20.pk'),
        ('trigger_50_pattern_40', '{}_closest_trigger_50_pattern_40.pk'),
        ('trigger_50_pattern_60', '{}_closest_trigger_50_pattern_60.pk'),
        ('trigger_50_pattern_80', '{}_closest_trigger_50_pattern_80.pk'),
        ('trigger_50_pattern_100', '{}_closest_trigger_50_pattern_100.pk'),
        ('trigger_50_pattern_120', '{}_closest_trigger_50_pattern_120.pk'),
        ('trigger_50_pattern_140', '{}_closest_trigger_50_pattern_140.pk'),
        ('trigger_50_pattern_160', '{}_closest_trigger_50_pattern_160.pk'),
        ('trigger_50_pattern_180', '{}_closest_trigger_50_pattern_180.pk'),
        ('trigger_50_pattern_200', '{}_closest_trigger_50_pattern_200.pk'),
        # Compare with using 50 triggers without patterns.
        # Output 10 same points to draw the figures.
        ('c1_trigger_50', '{}_closest_trigger_50.pk'),
        ('c2_trigger_50', '{}_closest_trigger_50.pk'),
        ('c3_trigger_50', '{}_closest_trigger_50.pk'),
        ('c4_trigger_50', '{}_closest_trigger_50.pk'),
        ('c5_trigger_50', '{}_closest_trigger_50.pk'),
        ('c6_trigger_50', '{}_closest_trigger_50.pk'),
        ('c7_trigger_50', '{}_closest_trigger_50.pk'),
        ('c8_trigger_50', '{}_closest_trigger_50.pk'),
        ('c9_trigger_50', '{}_closest_trigger_50.pk'),
        ('c10_trigger_50', '{}_closest_trigger_50.pk'),
    ]
}


def compute_scores(relation, exp):
    scores = []
    for name, pk in expriments[exp]:
        pk = pk.format(relation)
        pk_file = os.path.join(folder, pk)
    
        prev_precision, prev_recall, prev_fscore, prev_spec = 0, 0, 0, 0
        with open(pk_file, 'rb') as f:
            precision, recall, average_precision, thresholds = pickle.load(f)
            for i in range(len(recall[0])):
    
                if recall[0][i] == 0 or precision[0][i] == 0:
                    continue
    
                specificity = 0

                # There are some positive instances lost during generation
                # of the test data for PLOC. Here we adjust (lower) the
                # recall to reflect this loss.
                # Also compute specificity.
                if relation == 'ploc':
                    recall[0][i] = recall[0][i] * 125 / 150
                    specificity = (1783 - (1 - precision[0][i]) * 125) / 1783
    
                if relation == 'mirgene':
                    specificity = (775 - (1 - precision[0][i]) * 465) / 775
    
                if relation == 'ppi':
                    specificity = (4611 - (1 - precision[0][i]) * 1000) / 4611
    
                fscore = (precision[0][i]*recall[0][i]*2) / (precision[0][i]+recall[0][i])
    
                if thresholds[0][i] < 0.5:
                    # Save scores for threshold < 0.5.
                    prev_precision = precision[0][i]
                    prev_recall = recall[0][i]
                    prev_fscore = fscore
                    prev_spec = specificity
                else:
                    # Print scores at threshold 0.5.
                    scores.append((relation, name, prev_precision, prev_recall, prev_fscore, prev_spec))
                    break
    return scores


def compute_scores_at_30_recall(relation, exp):
    scores = []
    for name, pk in expriments[exp]:
        pk = pk.format(relation)
        pk_file = os.path.join(folder, pk)

        prev_precision, prev_recall, prev_fscore, prev_spec = 0, 0, 0, 0
        with open(pk_file, 'rb') as f:
            precision, recall, average_precision, thresholds = pickle.load(f)
            for i in range(len(recall[0])):

                if recall[0][i] == 0 or precision[0][i] == 0:
                    continue

                specificity = 0

                # There are some positive instances lost during generation
                # of the test data for PLOC. Here we adjust (lower) the
                # recall to reflect this loss.
                # Also compute specificity.
                if relation == 'ploc':
                    recall[0][i] = recall[0][i] * 125 / 150
                    specificity = (1783 - (1 - precision[0][i]) * 125) / 1783

                if relation == 'mirgene':
                    specificity = (775 - (1 - precision[0][i]) * 465) / 775

                if relation == 'ppi':
                    specificity = (4611 - (1 - precision[0][i]) * 1000) / 4611

                fscore = (precision[0][i] * recall[0][i] * 2) / (
                precision[0][i] + recall[0][i])

                if recall[0][i] > 0.3:
                    # Save scores for threshold < 0.5.
                    prev_precision = precision[0][i]
                    prev_recall = recall[0][i]
                    prev_fscore = fscore
                    prev_spec = specificity
                else:
                    # Print scores at threshold 0.5.
                    scores.append((relation, name, prev_precision, prev_recall,
                                   prev_fscore, prev_spec))
                    break
    return scores


if __name__ == '__main__':
    # Scores in Table 5.
    for relation in ['mirgene', 'ppi', 'ploc']:
        scores = compute_scores(relation, 'basic')
        for relation, name, p, r, f, s in scores:
            print(relation, name, p, r, f, s)
        scores = compute_scores_at_30_recall(relation, 'basic')
        for relation, name, p, r, f, s in scores:
            print(relation+'_30_recall', name, p, r, f, s)