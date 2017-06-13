#!/usr/bin/env bash
PYTHONPATH=src/ python src/multi_instance.py data/train/ppi.txt data/test/ppi.txt eval/pr_points/ppi_mi.pk R_PPI
PYTHONPATH=src/ python src/multi_instance.py data/train/mirgene.txt data/test/mirgene.txt eval/pr_points/mirgene_mi.pk R_MIRTAR
PYTHONPATH=src/ python src/multi_instance.py data/train/ploc.txt data/test/ploc.txt eval/pr_points/ploc_mi.pk R_PLOC

