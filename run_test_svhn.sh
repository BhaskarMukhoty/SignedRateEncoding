#!/usr/bin/bash

dataset='svhn'
dev=2
attack_mode='bptr' #'bptt'

declare -a arr=("vgg11_rate_signed_False_T4_clean" "vgg11_rate_signed_False_T4_rate_GN[0.007843][bptt]" "vgg11_rate_signed_False_T4_rate_FGSM[0.007843][bptt]" "vgg11_rate_signed_False_T4_rate_PGD[0.007843][bptt]" "vgg11_rate_signed_False_T4_rate_SEA[2.000000][bptt]")
for model_name in "${arr[@]}"
do
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}"
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --attack gn --eps 8 
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --attack fgsm --eps 8 --attack_mode "${attack_mode}" 
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --attack pgd --eps 8 --alpha 2 --steps 8 --attack_mode "${attack_mode}"
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --attack sea --eps 10
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --attack sea --eps 20
done 

declare -a arr=("vgg11_rate_signed_True_T4_clean" "vgg11_rate_signed_True_T4_rate_GN[0.007843][bptt]" "vgg11_rate_signed_True_T4_rate_FGSM[0.007843][bptt]" "vgg11_rate_signed_True_T4_rate_PGD[0.007843][bptt]" "vgg11_rate_signed_True_T4_rate_SEA[5.000000][bptt]")

for model_name in "${arr[@]}"
do
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --signed
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --attack gn --eps 8 --signed
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --attack fgsm --eps 8 --signed --attack_mode "${attack_mode}"
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --attack pgd --eps 8 --alpha 2 --steps 8 --signed --attack_mode "${attack_mode}"
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --attack sea --eps 10 --signed
    python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --attack sea --eps 20 --signed

done
