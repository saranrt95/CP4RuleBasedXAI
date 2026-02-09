#!/bin/bash
for dataset in $(echo 'breastW' 'liver' 'spambase' 'p2p' 'ssh' 'platooning'); do 
	for relevance in $(echo 'true' 'false');do
        for similarity in $(echo 'true');do
		    (python3 main.py $dataset $relevance $similarity)
        done
	done
done
