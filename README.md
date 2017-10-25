# Paper

[Noise Reduction Methods for Distantly Supervised Biomedical Relation Extraction](http://aclweb.org/anthology/W17-2323)

Please cite the paper if you use the codes:
<pre>
@INPROCEEDINGS{Li2017-zu,
  title           = "Noise Reduction Methods for Distantly Supervised
                     Biomedical Relation Extraction",
  booktitle       = "Proceedings of the {BioNLP} 2017 workshop",
  author          = "Li, Gang and Wu, Cathy H and Vijay-Shanker, K",
  pages           = "184--193",
  year            =  2017,
  conference      = "BioNLP 2017"
}
</pre>


## Requirements
* python 2.7 (should work with 3.6 but not tested)
* bash terminal

## Setup
* Download data: `wget http://biotm.cis.udel.edu/biotm/projects/ds/data.tar.gz`
* Uncompress data folder: `tar -zxvf data.tar.gz`
* Create a python virtual environment: `virtualenv env`
* Activate virtual environment: `. env/bin/activate`
* Install python modules: `pip install -r python_modules.txt`

## Run experiments

This is optional as all the result files are already included in this repo. Run the experiments if you want to regenerate them.

First activate the virtual environment: `. env/bin/activate`

### Baseline and all the heuristics
`sh script/single_instance.sh`

### Multi-instance learning baseline
`sh script/multi_instance.sh`

## Evaluation results and figures

Figures are stored in eval/figures

### Compute scores in Table 5
`python src/compute_scores.py`

### Draw Figure 2
`python src/draw_scores.py`

### Draw Figure 3
`python src/draw_curves.py`
