# SignedRateEncoding
Code corresponding to the paper (https://openreview.net/forum?id=qLh6Ufvnuc)

**IMPROVING GENERALIZATION AND ROBUSTNESS IN SNNS THROUGH SIGNED RATE ENCODING AND SPARSE ENCODING ATTACKS**
Bhaskar Mukhoty, Hilal AlQuabeh, Bin Gu


## Requirements
The conda environment is supplied in requirements.txt

## Usage 
Signed Rate Encoding (--signed) 

#### Training CIFAR-10/100
python main_train.py --dataset cifar10  --signed

python main_train.py --dataset cifar10 --signed --attack gn --eps 8

python main_train.py --dataset cifar10 --signed --attack fgsm --eps 8 

python main_train.py --dataset cifar10 --signed --attack pgd --eps 8 --alpha 2 --steps 4

python main_train.py --dataset cifar10 --signed --attack sea --eps 5



#### Training SVHN
python main_train.py --dataset svhn --signed

python main_train.py --dataset svhn --signed --attack gn --eps 2 

python main_train.py --dataset svhn --signed --attack fgsm --eps 2

python main_train.py --dataset svhn --signed --attack pgd --eps 2 --alpha 1 --steps 2

python main_train.py --dataset svhn --signed --attack sea --eps 5      # Rate encoding does not converge with --eps 5, use --eps 2


#### Training Imagenet-100
python main_train.py --dataset imagenet100 --signed --model sewresnet 

#### Testing
The testing usage with hyper-parameters are supplied in run_test_{dataset}.sh
