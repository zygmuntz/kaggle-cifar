#!/bin/bash

#--------------------------
# input params:
# $1 gpu
# $2 training set
#-------------------------

##--------------------------
##  trian nets
##--------------------------
#if [ $2 -eq 0 ]
#then
#    for i in 01 02 03 04 
#    do
#        ./train_nn-dcf.sh $1 run$i
#    done
#elif [ $2 -eq 1 ]
#then
#    for i in 05 06 07 08 
#    do
#        ./train_nn-dcf.sh $1 run$i
#    done
#else
#    for i in 09 10 11 12
#    do
#        ./train_nn-dcf.sh $1 run$i
#    done
#fi

#------------------------
# combine results
#------------------------
output_file=model_fc128-dcf-$2/combine_mean_log.txt
#echo 'combine net' > $output_file
for i in 01 02 03 04 05 06 07 08 09 10 11 12
do
    time optirun --no-xorg python ./shownet.py -f \
        ./model_fc128-dcf-$2/model_fc128-dcf-$2_run${i} \
        --gpu=0 --write-mv-result=./model_fc128-dcf-$2/run${i}_pred_mean.mat \
        | tee $output_file
done
time python ./combine_pred.py ./model_fc128-dcf-$2/*mean.mat | tee $output_file


