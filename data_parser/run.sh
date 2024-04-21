#!/bin/bash
N=6
dataset=UBFC1
if [ -n "$1" ];then
    dataset=$1
fi
cd /home/a406/Source/signal_vector/
files=`./data_loader/get_dataset_files.sh $dataset`
for i in $files; do
    (   
        echo $i
        python ./data_parser/get_raw.py $i
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

echo "all done"