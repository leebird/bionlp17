#!/usr/bin/env bash


for RELATION in mirgene ppi ploc
do
	# Baseline.
	PYTHONPATH=src/ python src/single_instance.py \
		data/train/${RELATION}.txt \
		data/test/${RELATION}.txt \
	    eval/pr_points/${RELATION}.pk \
		none

	# Negatives from raw distantly labeled data.
	# Used to combine with filtered positive data later.
	grep 'NONE' data/train/${RELATION}.txt > data/train/negative.txt

	# Closest pairs.
	# rel_closest.txt only contains positive, combine it with
	# negative from raw distantly labeled data.
	cat data/train/negative.txt \
        data/train/${RELATION}_closest.txt > data/train/train.txt
	PYTHONPATH=src/ python src/single_instance.py \
		data/train/train.txt \
		data/test/${RELATION}.txt \
	    eval/pr_points/${RELATION}_closest.pk \
		none

	# Min dep path frequency.
	PYTHONPATH=src/ python src/single_instance.py \
		data/train/${RELATION}.txt \
		data/test/${RELATION}.txt \
	    eval/pr_points/${RELATION}_mindepfreq_ \
		min_pos_dep_path_freq

	# Using 10-100 trigger words.
	# Only positive exists in those xxx_closest_trigger_xx.txt files.
	# We need to combine negatives from the raw data with them.
	for TRIGGER in 10 20 30 40 50 60 70 80 90 100
	do
		cat data/train/negative.txt \
	        data/train/${RELATION}_closest_trigger_${TRIGGER}.txt > data/train/train.txt
		PYTHONPATH=src/ python src/single_instance.py \
		data/train/train.txt \
		data/test/${RELATION}.txt \
	    eval/pr_points/${RELATION}_closest_trigger_${TRIGGER}.pk \
		none
	done

	# 100 patterns, 10-100 triggers
	for TRIGGER in 10 20 30 40 50 60 70 80 90 100
	do
		cat data/train/negative.txt \
	        data/train/${RELATION}_closest_trigger_${TRIGGER}.txt > data/train/train.txt

		PYTHONPATH=src/ python src/filter_negative.py \
			  data/train/train.txt \
			  data/train/train_filtered.txt \
			  100 ${RELATION} ${TRIGGER}

		PYTHONPATH=src/ python src/single_instance.py \
			  data/train/train_filtered.txt \
			  data/test/${RELATION}.txt \
			  eval/pr_points/${RELATION}_closest_trigger_${TRIGGER}_pattern_100.pk \
			  none
    done

	# 50 triggers, 20-200 patterns.
	grep 'NONE' data/train/${RELATION}.txt > data/train/negative.txt
	cat data/train/negative.txt \
        data/train/${RELATION}_closest_trigger_50.txt > data/train/train.txt

    for TOP_PATTERN in 20 40 60 80 100 120 140 160 180 200
    do
		PYTHONPATH=src/ python src/filter_negative.py \
			  data/train/train.txt \
			  data/train/train_filtered.txt \
			  ${TOP_PATTERN} ${RELATION} 50

		PYTHONPATH=src/ python src/single_instance.py \
			  data/train/train_filtered.txt \
			  data/test/${RELATION}.txt \
			  eval/pr_points/${RELATION}_closest_trigger_50_pattern_${TOP_PATTERN}.pk \
			  none
    done
done