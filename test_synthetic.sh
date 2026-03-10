#!/bin/bash
for dataset in $(echo 'easy' 'medium' 'hard'); do 
	for relevance in $(echo 'true');do
        for similarity in $(echo 'false');do
			for normalization in $(echo 'sigmoid' 'tanh' 'atan' 'decay'); do
		    (python3 main.py $dataset $relevance $similarity $normalization)
			done
        done
	done
done