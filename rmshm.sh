echo $(lsof /dev/shm | grep rn_norel | head -n 2 | tail -n 1 | awk '{print $2}' )
kill -2 $(lsof /dev/shm | grep rn_norel | head -n 2 | tail -n 1 | awk '{print $2}')
sleep 1
ls /dev/shm/rn*
