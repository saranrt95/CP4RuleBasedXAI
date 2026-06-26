#!/bin/bash
for dataset in $(echo 'medium'); do 
	for relevance in $(echo 'true');do
        for similarity in $(echo 'false');do
			for normalization in $(echo 'sigmoid'); do
		    (python3 main.py $dataset $relevance $similarity $normalization)
			done
        done
	done
done