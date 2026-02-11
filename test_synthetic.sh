#!/bin/bash
for dataset in $(echo 'medium'); do 
	for relevance in $(echo 'true');do
        for similarity in $(echo 'true');do
		    (python3 main.py $dataset $relevance $similarity)
        done
	done
done