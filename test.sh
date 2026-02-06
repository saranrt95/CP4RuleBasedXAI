#!/bin/bash
for dataset in $(echo 'breastW' 'liver' 'spambase'); do 
	for relevance in $(echo 'true' 'false');do
        for similarity in $(echo 'true' 'false');do
		    (python3 main.py $dataset $relevance $similarity)
        done
	done
done
