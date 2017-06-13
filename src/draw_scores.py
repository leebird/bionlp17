from __future__ import unicode_literals, print_function, division
import os
import matplotlib.pyplot as plt
from compute_scores import compute_scores


# Draw figures in Figure 2.

for rel in ['mirgene', 'ppi', 'ploc']:
    # Heuristic of trigger words.
    scores = compute_scores(rel, 'h2')
    precisions = [s[2] for s in scores]
    recalls = [s[3] for s in scores]
    fscores = [s[4] for s in scores]

    prec_color, recall_color, fscore_color = 'black', 'black', 'black'

    x_axis = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    ]


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('# of trigger stems', fontsize=24)

    ax1.plot(x_axis, precisions[:10], marker='o', color=prec_color,
             markersize=15, lw=2, linestyle='-', fillstyle='none')
    ax1.plot(x_axis, recalls[:10], marker='s', color=recall_color,
             markersize=15, lw=2, linestyle='-', fillstyle='none')
    ax1.plot(x_axis, fscores[:10], marker='^', color=fscore_color,
             markersize=15, lw=2, linestyle='-', fillstyle='none')

    ax1.plot(x_axis, precisions[10:], color=prec_color,
             markersize=15, lw=2, marker='o', linestyle='--', fillstyle='none')
    ax1.plot(x_axis, recalls[10:], color=recall_color,
             markersize=15, lw=2, marker='s', linestyle='--', fillstyle='none')
    ax1.plot(x_axis, fscores[10:], color=fscore_color,
             markersize=15, lw=2, marker='^', linestyle='--', fillstyle='none')


    plt.tick_params(labelsize=20)
    # Fig 2. a-c.
    figfile = os.path.join('eval/figures/{}_trigger.png'.format(rel))
    fig.savefig(figfile, dpi=300, pad_inches=0 ,bbox_inches='tight')
    plt.show()

    # High-confidence patterns.
    scores = compute_scores(rel, 'h3')
    precisions = [s[2] for s in scores]
    recalls = [s[3] for s in scores]
    fscores = [s[4] for s in scores]

    x_axis = [
        20, 40, 60, 80, 100, 120, 140, 160, 180, 200
    ]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('# of patterns', fontsize=24)

    ax1.plot(x_axis, precisions[:10], marker='o', color=prec_color,
             markersize=15, lw=2, linestyle='-', fillstyle='none')
    ax1.plot(x_axis, recalls[:10], marker='s', color=recall_color,
             markersize=15, lw=2, linestyle='-', fillstyle='none')
    ax1.plot(x_axis, fscores[:10], marker='^', color=fscore_color,
             markersize=15, lw=2, linestyle='-', fillstyle='none')

    ax1.plot(x_axis, precisions[10:], color=prec_color, markersize=15,
             lw=2, marker='o', linestyle='--', fillstyle='none')
    ax1.plot(x_axis, recalls[10:], color=recall_color, markersize=15,
             lw=2, marker='s', linestyle='--', fillstyle='none')
    ax1.plot(x_axis, fscores[10:], color=fscore_color, markersize=15,
             lw=2, marker='^', linestyle='--', fillstyle='none')


    plt.tick_params(labelsize=20)
    # Fig. 2 d-f.
    figfile = os.path.join('eval/figures/{}_pattern.png'.format(rel))
    fig.savefig(figfile, dpi=300, pad_inches=0 ,bbox_inches='tight')
    plt.show()

