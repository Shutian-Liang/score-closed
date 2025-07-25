#!/bin/bash
for data in "50k"; do 
    (nohup python -u main.py --sde_version vp --start sub_noise \
        --times 50 --device 4 --dataset $data > logfiles/sub_noise_$data.log 2>&1 )& \
    
    (nohup python -u main.py --sde_version vp --start noise \
        --times 50 --device 3 --dataset $data > logfiles/noise_$data.log 2>&1 )& \
    
    wait;

done