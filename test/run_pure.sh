#!/bin/bash
N=6
files=`ls -d /tank/数据集/PURE/*|grep -E "PURE/[0-9]{2}-[0-9]{2}.json$"`
for i in $files; do
    (   
        python /home/a406/Source/signal_vector/test/test.py $i
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