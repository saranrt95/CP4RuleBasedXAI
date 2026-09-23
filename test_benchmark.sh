#!/bin/bash
for dataset in $(echo 'breastw' 'liver' 'spambase' 'rul_stride45'); do # 
	for normalization in $(echo 'sigmoid'); do 
		for beta in $(echo 0.01 0.1 1.0 10); do
			python3 main.py --dataset "$dataset" \
			 --use_relevance true \
			 --use_similarity false \
			 --normalization $normalization \
			 --beta "$beta"
		done
	done
done
