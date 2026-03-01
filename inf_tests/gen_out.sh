#!/bin/bash
N=50
dataset=ALL
if [ -n "$1" ];then
    dataset=$1
fi
files=`./data_loader/get_dataset_files.sh $dataset`
for i in $files; do
    (   
        python ./inf_tests/get_inf.py $i 
    ) &
    
    # allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi

done

# no more jobs to be started but wait for pending jobs
# (all need to be finished)
wait