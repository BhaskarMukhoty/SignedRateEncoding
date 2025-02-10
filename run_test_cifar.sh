#!/usr/bin/bash

#dataset='cifar10'
dataset='cifar100'
dev=1
#attack_mode='bptt'
attack_mode='bptr'
T=4
m=10



declare -a arr=("vgg11_rate_signed_False_T4_clean" "vgg11_rate_signed_False_T4_rate_GN[0.031373][bptt]" "vgg11_rate_signed_False_T4_rate_FGSM[0.031373][bptt]" "vgg11_rate_signed_False_T4_rate_PGD[0.031373][bptt]" "vgg11_rate_signed_False_T4_rate_SEA[5.000000][bptt]")

for model_name in "${arr[@]}"
do
    python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --time ${T} --m ${m}
    python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --time ${T} --m ${m} --attack gn --eps 8  
    python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --time ${T} --m ${m} --attack_mode "${attack_mode}" --attack fgsm --eps 8 
    python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --time ${T} --m ${m} --attack_mode "${attack_mode}" --attack pgd --eps 8 --alpha 2 --steps 8
    python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --time ${T} --attack_mode "${attack_mode}" --attack sea --eps 10
    python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --time ${T} --attack_mode "${attack_mode}" --attack sea --eps 20
done 

declare -a arr=("vgg11_rate_signed_True_T4_clean" "vgg11_rate_signed_True_T4_rate_GN[0.031373][bptt]" "vgg11_rate_signed_True_T4_rate_FGSM[0.031373][bptt]" "vgg11_rate_signed_True_T4_rate_PGD[0.031373][bptt]" "vgg11_rate_signed_True_T4_rate_SEA[5.000000][bptt]")

 for model_name in "${arr[@]}"
 do
     python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --signed --time ${T} --m ${m}
     python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --signed --time ${T} --m ${m} --attack gn --eps 8 
     python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --signed --time ${T} --m ${m} --attack_mode "${attack_mode}" --attack fgsm --eps 8 
     python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --signed --time ${T} --m ${m} --attack_mode "${attack_mode}" --attack pgd --eps 8 --alpha 2 --steps 8
     python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --signed --time ${T} --attack_mode "${attack_mode}" --attack sea --eps 10 
     python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --signed --time ${T} --attack_mode "${attack_mode}" --attack sea --eps 20
done 
