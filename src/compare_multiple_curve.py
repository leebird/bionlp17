from __future__ import unicode_literals, print_function, division
import pickle
from itertools import cycle
import os
from collections import OrderedDict, defaultdict
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


rel = 'ppi'
folder = 'eval'

output_choices = {
    'basic': [
        ('baseline 1', '{}.pk'.format(rel)),
        ('DPFreq', '{}_mindepfreq_5.pk'.format(rel)),
        ('MI', '{}_mi.pk'.format(rel)),
        ('H1', '{}_closest.pk'.format(rel)),
        ('H2', '{}_closest_trigger_50.pk'.format(rel)),
        ('H3', '{}_closest_trigger_50_pattern_100.pk'.format(rel)),
    ]
}

exps = [
    ('baseline 1', '{}.pk'.format(rel)),
    ('DPFreq', '{}_mindepfreq_5.pk'.format(rel)),
    ('MI', '{}_mi_sen_pos_.pk'.format(rel)),
    #
    ('H1', '{}_closest.pk'.format(rel)),
    ('H2', '{}_sen_pos_clo_50.pk'.format(rel)),
    ('H3', '{}_50_filtered_100.pk'.format(rel)),

    # ('1', '{}_{}_sim_maxdeplen_1.pk'.format(rel, corpus)),
    # ('2', '{}_{}_sim_maxdeplen_2.pk'.format(rel, corpus)),
    # ('3', '{}_{}_sim_maxdeplen_3.pk'.format(rel, corpus)),
    # ('4', '{}_{}_sim_maxdeplen_4.pk'.format(rel, corpus)),
    # ('5', '{}_{}_sim_maxdeplen_5.pk'.format(rel, corpus)),
    # ('6', '{}_{}_sim_maxdeplen_6.pk'.format(rel, corpus)),
    # ('7', '{}_{}_sim_maxdeplen_7.pk'.format(rel, corpus)),
    # ('8', '{}_{}_sim_maxdeplen_8.pk'.format(rel, corpus)),
    # ('9', '{}_{}_sim_maxdeplen_9.pk'.format(rel, corpus)),

    # ('2', '{}_{}_sim_mindepfreq_2.pk'.format(rel, corpus)),
    # ('3', '{}_{}_sim_mindepfreq_3.pk'.format(rel, corpus)),
    # ('4', '{}_{}_sim_mindepfreq_4.pk'.format(rel, corpus)),
    # ('5', '{}_{}_sim_mindepfreq_4.pk'.format(rel, corpus)),
    # ('6', '{}_{}_sim_mindepfreq_6.pk'.format(rel, corpus)),
    # ('7', '{}_{}_sim_mindepfreq_7.pk'.format(rel, corpus)),
    # ('8', '{}_{}_sim_mindepfreq_8.pk'.format(rel, corpus)),
    # ('9', '{}_{}_sim_mindepfreq_9.pk'.format(rel, corpus)),

    # ('H2 (10)', '{}_{}_{}_10.pk'.format(rel, corpus, filtering)),
    # ('H2 (20)', '{}_{}_{}_20.pk'.format(rel, corpus, filtering)),
    # ('H2 (30)', '{}_{}_{}_30.pk'.format(rel, corpus, filtering)),
    # ('H2 (40)', '{}_{}_{}_40.pk'.format(rel, corpus, filtering)),
    # ('H2 (50)', '{}_{}_{}_50.pk'.format(rel, corpus, filtering)),
    # ('H2 (60)', '{}_{}_{}_60.pk'.format(rel, corpus, filtering)),
    # ('H2 (70)', '{}_{}_{}_70.pk'.format(rel, corpus, filtering)),
    # ('H2 (80)', '{}_{}_{}_80.pk'.format(rel, corpus, filtering)),
    # ('H2 (90)', '{}_{}_{}_90.pk'.format(rel, corpus, filtering)),
    # ('H2 (100)', '{}_{}_{}_100.pk'.format(rel, corpus, filtering)),

    # ('10', '{}_{}_{}_filtered_10.pk'.format(rel, corpus, filtering)),
    # ('H3 (20)', '{}_{}_{}_filtered_20.pk'.format(rel, corpus, filtering)),
    # ('30', '{}_{}_{}_filtered_30.pk'.format(rel, corpus, filtering)),
    # ('H3 (40)', '{}_{}_{}_filtered_40.pk'.format(rel, corpus, filtering)),
    # ('50', '{}_{}_{}_filtered_50.pk'.format(rel, corpus, filtering)),
    # ('H3 (60)', '{}_{}_{}_filtered_60.pk'.format(rel, corpus, filtering)),
    # ('70', '{}_{}_{}_filtered_70.pk'.format(rel, corpus, filtering)),
    # ('H3 (80)', '{}_{}_{}_filtered_80.pk'.format(rel, corpus, filtering)),
    # ('90', '{}_{}_{}_filtered_90.pk'.format(rel, corpus, filtering)),
    # ('H3 (100)', '{}_{}_{}_filtered_100.pk'.format(rel, corpus, filtering)),

    # ('110', '{}_{}_{}_filtered_110.pk'.format(rel, corpus, filtering)),
    # ('H3 (120)', '{}_{}_{}_filtered_120.pk'.format(rel, corpus, filtering)),
    # ('130', '{}_{}_{}_filtered_130.pk'.format(rel, corpus, filtering)),
    # ('H3 (140)', '{}_{}_{}_filtered_140.pk'.format(rel, corpus, filtering)),
    # ('150', '{}_{}_{}_filtered_150.pk'.format(rel, corpus, filtering)),
    # ('H3 (160)', '{}_{}_{}_filtered_160.pk'.format(rel, corpus, filtering)),
    # ('170', '{}_{}_{}_filtered_170.pk'.format(rel, corpus, filtering)),
    # ('H3 (180)', '{}_{}_{}_filtered_180.pk'.format(rel, corpus, filtering)),
    # ('190', '{}_{}_{}_filtered_190.pk'.format(rel, corpus, filtering)),
    # ('H3 (200)', '{}_{}_{}_filtered_200.pk'.format(rel, corpus, filtering)),

    #('H2', '{}_{}_sim_mindepfreq_4.pk'.format(rel, corpus)),
    #('H3', '{}_{}_sim_negsam_0.9.pk'.format(rel, corpus)),
    # ('H4', '{}_{}_sen_pos_30.pk'.format(rel, corpus)),
    # ('H5', '{}_{}_closest.pk'.format(rel, corpus)),
    # ('H4+5', '{}_{}_sen_pos_clo_70.pk'.format(rel, corpus)),
    # ('H6', '{}_{}_sen_pos_clo_70_filtered_80.pk'.format(rel, corpus)),

    # ('10', rel+'_{}_{}_filtered_10.pk'.format(corpus, filtering)),
    # ('20', rel+'_{}_{}_filtered_20.pk'.format(corpus, filtering)),
    # ('30', rel+'_{}_{}_filtered_30.pk'.format(corpus, filtering)),
    # ('40', rel+'_{}_{}_filtered_40.pk'.format(corpus, filtering)),
    # ('50', rel+'_{}_{}_filtered_50.pk'.format(corpus, filtering)),
    # ('60', rel+'_{}_{}_filtered_60.pk'.format(corpus, filtering)),
    # ('70', rel+'_{}_{}_filtered_70.pk'.format(corpus, filtering)),
    # ('80', rel+'_{}_{}_filtered_80.pk'.format(corpus, filtering)),
    # ('90', rel+'_{}_{}_filtered_90.pk'.format(corpus, filtering)),
    # ('100', rel+'_{}_{}_filtered_100.pk'.format(corpus, filtering)),
    # ('110', rel+'_{}_{}_filtered_110.pk'.format(corpus, filtering)),
    # ('120', rel+'_{}_{}_filtered_120.pk'.format(corpus, filtering)),
    # ('130', rel+'_{}_{}_filtered_130.pk'.format(corpus, filtering)),
    # ('140', rel+'_{}_{}_filtered_140.pk'.format(corpus, filtering)),
    # ('150', rel+'_{}_{}_filtered_150.pk'.format(corpus, filtering)),
    # ('160', rel+'_{}_{}_filtered_160.pk'.format(corpus, filtering)),
    # ('170', rel+'_{}_{}_filtered_170.pk'.format(corpus, filtering)),
    # ('180', rel+'_{}_{}_filtered_180.pk'.format(corpus, filtering)),
    # ('190', rel+'_{}_{}_filtered_190.pk'.format(corpus, filtering)),
    # ('200', rel+'_{}_{}_filtered_200.pk'.format(corpus, filtering)),

    # ('0.9', rel+'_{}_{}_0.9.pk'.format(corpus, filtering)),
    # ('sen', rel+'_{}_sen_0.8.pk'.format(corpus, filtering)),
    # ('sen_pos', rel+'_{}_sen_pos_0.8.pk'.format(corpus, filtering)),
    # ('sen_pos_clo', rel+'_filtered.pk'.format(corpus, filtering)),
    # ('sen_pos_clo_200', rel+'_{}_sen_pos_clo_0.8_filtered_200.pk'.format(corpus, filtering)),
    # ('20', rel+'_{}_{}_20.pk'.format(corpus, filtering)),
    # ('30', rel+'_{}_{}_30.pk'.format(corpus, filtering)),
    # ('40', rel+'_{}_{}_40.pk'.format(corpus, filtering)),
    # ('50', rel+'_{}_{}_50.pk'.format(corpus, filtering)),
    # ('sen', rel+'_{}_sen_gold.pk'.format(corpus)),
    # ('sen_pos', rel+'_{}_sen_pos_gold.pk'.format(corpus)),
    # ('sen_pos_clo', rel+'_{}_sen_pos_clo_gold.pk'.format(corpus)),
    # ('sen_pos_clo_0.9', rel+'_{}_sen_pos_clo_0.9.pk'.format(corpus)),
    # ('sen_pos_clo_0.8', rel+'_{}_sen_pos_clo_0.8.pk'.format(corpus)),
    # ('sen_pos_clo_0.85', rel+'_{}_sen_pos_clo_0.85.pk'.format(corpus)),
    # ('sen_pos_clo_filtered', rel+'_{}_sen_pos_clo_filtered.pk'.format(corpus)),
    # ('sen_pos_clo_gold', rel+'_{}_sen_pos_clo_gold.pk'.format(corpus)),
    # ('sen_pos_clo_20', rel+'_{}_sen_pos_clo_gold_filter_20.pk'.format(corpus)),
    # ('sen_pos_clo_40', rel+'_{}_sen_pos_clo_gold_filter_40.pk'.format(corpus)),
    # ('sen_pos_clo_60', rel+'_{}_sen_pos_clo_gold_filter_60.pk'.format(corpus)),
    # ('sen_pos_clo_80', rel+'_{}_sen_pos_clo_gold_filter_80.pk'.format(corpus)),
    # ('sen_pos_clo_100', rel+'_{}_sen_pos_clo_gold_filter_100.pk'.format(corpus)),
    # ('80', 'test2.pk'),
    # ('none', rel.lower()+'_dev_miml.pk'),
    # ('trigger', rel.lower()+'_dev_trigger_w2.pk'),
    # ('10', 'ploc_trigger_w2_top10.pk'),
    # ('trigger20', rel.lower()+'_trigger_w2_top20.pk'),
    # ('30', 'ploc_trigger_w2_top30.pk'),
    # ('trigger40', rel.lower()+'_trigger_w2_top40.pk'),
    # ('50', 'ploc_trigger_w2_top50.pk'),
    # ('trigger60', rel.lower()+'_trigger_w2_top60.pk'),
    # ('70', 'ploc_trigger_w2_top70.pk'),
    # ('80', rel.lower()+'_trigger_w2_top80.pk'),
    # ('90', 'ploc_trigger_w2_top90.pk'),
    # ('100', 'ploc_trigger_w2_top100.pk'),
]

name_to_scores = OrderedDict()
name_to_best_fscores = defaultdict(int)
name_to_best_scores = {}

for name, pk in exps:
    pk_file = os.path.join(folder, pk)

    with open(pk_file, 'rb') as f:
        precision, recall, average_precision, thresholds = pickle.load(f)
        for i in range(len(recall[0])):

            if recall[0][i] == 0 or precision[0][i] == 0:
                continue

            specificity = 0
            if rel == 'ploc':
                recall[0][i] = recall[0][i] * 125 / 150
                specificity = (1783 - (1 - precision[0][i]) * 125) / 1783

            if rel == 'mirtar':
                specificity = (775 - (1 - precision[0][i]) * 465) / 775

            if rel == 'ppi':
                specificity = (4611 - (1 - precision[0][i]) * 1000) / 4611

            fscore = recall[0][i]*precision[0][i]*2/(recall[0][i]+precision[0][i])
            if name_to_best_fscores[name] < fscore:
                name_to_best_fscores[name] = fscore
                name_to_best_scores[name] = (precision[0][i], recall[0][i], fscore, thresholds[0][i])
            # if 0.83 < thresholds[0][i] < 0.84:
            #     print('test', name, precision[0][i], recall[0][i], fscore, thresholds[0][i])

        name_to_scores[name] = (precision, recall, average_precision)

for name, scores in name_to_best_scores.items():
    print(name, '{} {} {} {}'.format(scores[0], scores[1], scores[2], scores[3]))

# Plot Precision-Recall curve
lw = 1
colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']

# Plot Precision-Recall curve for each class
plt.rc('xtick', labelsize=24) 
plt.rc('ytick', labelsize=24) 


plt.clf()
count = 0
for name, scores in name_to_scores.items():
    precision, recall, average_precision = scores
    if name == 'baseline 1':
        plt.plot(recall[0], precision[0], lw=2, label=name, color='black', marker='+', fillstyle='none', markevery=0.2, markersize=15)
    elif name == 'DPFreq':
        plt.plot(recall[0], precision[0], lw=2, label=name, color='black', marker='x', fillstyle='none', markevery=0.2, markersize=15)
    elif name == 'H1':
        plt.plot(recall[0], precision[0], lw=2, label=name, color='black', marker='s', fillstyle='none', markevery=0.2, markersize=15)
    elif name == 'H2':
        plt.plot(recall[0], precision[0], lw=2, label=name, color='black', marker='^', fillstyle='none', markevery=0.2, markersize=15)
    elif name == 'H3':
        plt.plot(recall[0], precision[0], lw=2, label=name, color='black', marker='o', fillstyle='none', markevery=0.2, markersize=15)
    elif name == 'MI':
        plt.plot(recall[0], precision[0], lw=2, label=name, color='black', marker='d', fillstyle='none', markevery=0.2, markersize=15)
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
# ax = fig.gca()
# ax.set_xticks(np.arange(0, 1, 0.1))
# ax.set_yticks(np.arange(0, 1., 0.1))
# plt.grid(True, linestyle='--')

#figfile = os.path.join(folder, rel+'_'+corpus+'_'+filtering+'.png')
figfile = os.path.join('/home/leebird/Desktop/acl17-latex', rel+'_'+corpus+'_'+filtering+'.png')
plt.savefig(figfile, dpi=300, pad_inches=0 ,bbox_inches='tight')
plt.show()
