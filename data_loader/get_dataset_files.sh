#!/bin/bash
if [[ $1 == "PURE" ]] || [[ $1 == "ALL" ]];then
    ls -d /tank/数据集/PURE/*|grep -E "PURE/[0-9]{2}-[0-9]{2}.json$"
fi
if [[ $1 == "ECG-Fitness" ]]|| [[ $1 == "ALL" ]];then
    find /tank/数据集/ECG-Fitness/视频版/|grep c920-1.avi
fi
if [[ $1 == "UBFC1" ]]|| [[ $1 == "ALL" ]];then
    find /tank/数据集/UBFC/UBFC1/|egrep "/vid.avi"
fi
if [[ $1 == "UBFC2" ]]|| [[ $1 == "ALL" ]];then
    find /tank/数据集/UBFC/UBFC2/|egrep "/vid.avi"
fi
if [[ $1 == "UBFC-Phys" ]]|| [[ $1 == "ALL" ]];then
    find /tank/数据集/UBFC-Phys_dataset/|egrep "/vid_[^/]*.avi"
fi

