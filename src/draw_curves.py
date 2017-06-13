from __future__ import print_function, division
import os
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt


# Draw Figure 3.

for rel in ['ppi', 'mirgene', 'ploc']:
    exps = [
        ('baseline 1', '{}.pk'.format(rel)),
        ('DPFreq', '{}_mindepfreq_5.pk'.format(rel)),
        ('MI', '{}_mi.pk'.format(rel)),
        ('H1', '{}_closest.pk'.format(rel)),
        ('H2', '{}_closest_trigger_50.pk'.format(rel)),
        ('H3', '{}_closest_trigger_50_pattern_100.pk'.format(rel)),
    ]

    name_to_scores = OrderedDict()
    for name, pk in exps:
        pk_file = os.path.join('eval/pr_points/', pk)

        with open(pk_file, 'rb') as f:
            precision, recall, average_precision, thresholds = pickle.load(f)
            for i in range(len(recall[0])):

                if recall[0][i] == 0 or precision[0][i] == 0:
                    continue

                # Adjust recall of PLOC as it missed some TP during
                # NER of subcellular location.
                if rel == 'ploc':
                    recall[0][i] = recall[0][i] * 125 / 150

            name_to_scores[name] = (precision, recall, average_precision)


    # Plot Precision-Recall curve for each class
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.clf()

    count = 0
    for name, scores in name_to_scores.items():
        precision, recall, average_precision = scores
        if name == 'baseline 1':
            plt.plot(recall[0], precision[0], lw=2, label=name, color='black',
                     marker='+', fillstyle='none', markevery=0.2, markersize=15)
        elif name == 'DPFreq':
            plt.plot(recall[0], precision[0], lw=2, label=name, color='black',
                     marker='x', fillstyle='none', markevery=0.2, markersize=15)
        elif name == 'H1':
            plt.plot(recall[0], precision[0], lw=2, label=name, color='black',
                     marker='s', fillstyle='none', markevery=0.2, markersize=15)
        elif name == 'H2':
            plt.plot(recall[0], precision[0], lw=2, label=name, color='black',
                     marker='^', fillstyle='none', markevery=0.2, markersize=15)
        elif name == 'H3':
            plt.plot(recall[0], precision[0], lw=2, label=name, color='black',
                     marker='o', fillstyle='none', markevery=0.2, markersize=15)
        elif name == 'MI':
            plt.plot(recall[0], precision[0], lw=2, label=name, color='black',
                     marker='d', fillstyle='none', markevery=0.2, markersize=15)
        else:
            continue
        count += 1


    plt.xlim([0.0, 1.0])
    plt.ylim([0, 1.0])

    plt.xlabel('Recall', fontsize=36)
    plt.ylabel('Precision', fontsize=36)
    plt.tick_params(labelsize=20)

    # plt.title('Precision-Recall curves for {} relation'.format(rel.upper()))
    # plt.legend(loc="upper right", prop={"size":8})

    # fig = plt.figure()

    figfile = os.path.join('eval/figures/{}_prcurve.png'.format(rel))
    plt.savefig(figfile, dpi=300, pad_inches=0 ,bbox_inches='tight')
    plt.show()
