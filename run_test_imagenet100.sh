#!/usr/bin/bash

dataset='imagenet100'
dev=1
model='sewresnet'
bs=128
attack_mode='bptt'

model_name="sewresnet_rate_signed_False_T4_clean"

 python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --model "${model}" --batch_size ${bs} --attack_mode "${attack_mode}"
 python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --model "${model}" --batch_size ${bs} --attack_mode "${attack_mode}" --attack gn --eps 8 
 python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --model "${model}" --batch_size ${bs} --attack_mode "${attack_mode}" --attack fgsm --eps 8 
 python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --model "${model}" --batch_size ${bs} --attack_mode "${attack_mode}" --attack pgd --eps 8 --alpha 2 --steps 8
 python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --model "${model}" --batch_size ${bs} --attack_mode "${attack_mode}" --attack sea --eps 10 
 python main_test.py --identifier "${model_name}"  --device ${dev} --dataset "${dataset}" --model "${model}" --batch_size ${bs} --attack_mode "${attack_mode}" --attack sea --eps 20


model_name="sewresnet_rate_signed_True_T4_clean"

python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --model "${model}" --batch_size ${bs} --signed --attack_mode "${attack_mode}"
python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --model "${model}" --batch_size ${bs} --signed --attack_mode "${attack_mode}" --attack gn --eps 8 
python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --model "${model}" --batch_size ${bs} --signed --attack_mode "${attack_mode}" --attack fgsm --eps 8
python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --model "${model}" --batch_size ${bs} --signed --attack_mode "${attack_mode}" --attack pgd --eps 8 --alpha 2 --steps 8
python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --model "${model}" --batch_size ${bs} --signed --attack_mode "${attack_mode}" --attack sea --eps 10
python main_test.py --identifier "${model_name}"  --device "${dev}" --dataset "${dataset}" --model "${model}" --batch_size ${bs} --signed --attack_mode "${attack_mode}" --attack sea --eps 20
