#!/bin/bash
for dataset in $(echo 'breastW' 'liver' 'spambase' 'p2p' 'ssh' 'platooning' 'diabetes'); do # 
	for relevance in $(echo 'true');do
        for similarity in $(echo 'false');do
			for normalization in $(echo 'sigmoid'); do # 'tanh' 'atan' 'decay'
		    (python3 main.py $dataset $relevance $similarity $normalization)
			done
        done
	done
done
