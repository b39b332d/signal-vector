#!/bin/bash
N=50
dataset=PURE
if [ -n "$1" ];then
    dataset=$1
fi
files=`./data_loader/get_dataset_files.sh $dataset|grep "/01"`
for i in $files; do
    (   
        python ./vec/get_vec.py $i |& awk '{print "'$i' " $0}'
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