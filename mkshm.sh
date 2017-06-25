#!/bin/sh
nohup /home/bgshin/virt/rn/bin/python -u /home/bgshin/works/rn/rn_shm_loader.py > rnsh.txt &
while [ ! -f /dev/shm/rn_rel_trn_0 ]
do
    sleep 1
done
ls /dev/shm/rn*
echo 'done'

