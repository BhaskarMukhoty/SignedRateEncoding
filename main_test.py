import argparse
import os
from models.VGG import *
from autoattack import AutoAttack
from functions import *
from utils import val, val2
import attack
import copy
import torch
import json
from data_loaders import cifar10, cifar100, imagenet100, svhn

parser = argparse.ArgumentParser()
parser.add_argument('-j','--workers',default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=128, type=int,metavar='N',help='mini-batch size')
parser.add_argument('-sd', '--seed',default=42,type=int,help='seed for initializing training.')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')
# model configuration
parser.add_argument('-data', '--dataset', default='cifar10',type=str,help='dataset')
parser.add_argument('-arch','--model', default='vgg11', type=str,help='model')
parser.add_argument('-T','--time', default=4, type=int, metavar='N',help='snn simulation time')
parser.add_argument('-id', '--identifier', type=str, help='model statedict identifier')
parser.add_argument('-config', '--config', default='', type=str,help='test configuration file')
# training configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')
# adv atk configuration
parser.add_argument('--attack',default='',type=str,help='attack')
parser.add_argument('-eps','--eps',default=8,type=float,metavar='N',help='attack eps')
parser.add_argument('-atk_m','--attack_mode',default='bptt', type=str,help='attack mode')
# only pgd
parser.add_argument('-alpha','--alpha',default=2,type=float,metavar='N',help='pgd attack alpha')
parser.add_argument('-steps','--steps',default=8,type=int,metavar='N',help='pgd attack steps')
parser.add_argument('-bb','--bbmodel',default='',type=str,help='black box model')
parser.add_argument('-enc','--encoding',default='rate',type=str,help='encoding')
parser.add_argument('-atk_enc','--atk_encoding',default='rate',type=str,help='attack encoding')
parser.add_argument('-ext','--ext',default='',type=str,help='external Path')
parser.add_argument("--cache_dataset", action="store_true", help="Cache the datasets for quicker initialization. It also serializes the transforms")
parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
parser.add_argument("--signed", action="store_true", help="Enable nromalization")
parser.add_argument('--m',default=1,type=int,help='no. of noisy samples')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    if args.dataset.lower() == 'cifar10':
        num_labels = 10
        train_dataset, val_dataset, znorm = cifar10(normalized = args.signed)
    elif args.dataset.lower() == 'svhn':
        num_labels = 10
        train_dataset, val_dataset, znorm = svhn(normalized = args.signed)
    elif args.dataset.lower() == 'cifar100':
        num_labels = 100
        train_dataset, val_dataset, znorm = cifar100(normalized = args.signed)
    elif args.dataset.lower() == 'imagenet100':
        num_labels = 100
        train_dataset, val_dataset, znorm = imagenet100(normalized = args.signed)

    log_dir = '%s-Results'% (args.dataset)
   
    model_dir = args.ext + '%s-checkpoints'% (args.dataset)
    
    model_dir = os.path.join(model_dir, args.model)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = get_logger(os.path.join(log_dir, '%s.log'%(str(args.identifier)+args.suffix)))
    logger.info(vars(args))
    logger.info('start testing!')

    seed_all(args.seed)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
    

    model = create_model(args.model.lower(), args.encoding, args.signed, args.atk_encoding, False, args.time, num_labels, znorm)
    
        
    checkpoint = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_simulation_time(args.time)
    model.to(device)
    #have bb model
    if len(args.bbmodel) > 0:
        bbmodel = copy.deepcopy(model)
        bbstate_dict = torch.load(os.path.join(model_dir, args.bbmodel+'.pth'), map_location=torch.device('cpu'))
        bbmodel.load_state_dict(bbstate_dict, strict=False)
    else:
        bbmodel = None

    if len(args.config) > 0:
        with open(args.config+'.json', 'r') as f:
            config = json.load(f)
    else:
        config = [{}]
    for atk_config in config:
        for arg in atk_config.keys():
            setattr(args, arg, atk_config[arg])
        if 'bb' in atk_config.keys() and atk_config['bb']:
            atkmodel = bbmodel
        else:
            atkmodel = model

        if args.attack_mode == 'bptt':
            ff = BPTT_attack
        elif args.attack_mode == 'bptr':
            ff = BPTR_attack
        else:
            ff = Act_attack
        
        print(f'Attack Mode: {ff}')
        
        #adversary.attacks_to_run = ['apgd-ce']
        #adversary.apgd.verbose = True

        
        if args.attack.lower() == 'sea':
            print(f'sea, model encoding:{model.encoding}, signed:{model.signed}, eps={args.eps}')
            atk = attack.SEA(atkmodel, device, forward_function=ff, eps=args.eps , T=args.time, signed=args.signed)
        elif args.attack.lower() == 'fgsm':
            print(f'FGSM, model encoding:{model.encoding} , attack encoding: {atkmodel.atk_encoding}, eps={args.eps}')
            atk = attack.FGSM(atkmodel, device, forward_function=ff, eps=args.eps / 255 , T=args.time, signed=args.signed, m=args.m)
        elif args.attack.lower() == 'pgd':
            print(f'PGD, model encoding:{model.encoding} , attack encoding: {atkmodel.atk_encoding}, eps={args.eps}')
            atk = attack.PGD(atkmodel, device, forward_function=ff, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, T=args.time, signed=args.signed, m=args.m)
        elif args.attack.lower() == 'gn':
            print(f'GN,  model encoding:{model.encoding} , attack encoding: {atkmodel.atk_encoding} , eps={args.eps}')
            atk = attack.GN(atkmodel, device, forward_function=ff, eps=args.eps / 255, T=args.time, signed = args.signed)
        elif args.attack.lower() == 'apgd_l1':
            print(f'APGD_L1,  model encoding:{model.encoding} , attack encoding: {atkmodel.atk_encoding} , eps={args.eps}')
            class model_1(nn.Module):
                def __init__(self, model):
                    super(model_1, self).__init__()
                    self.model = model
                def forward(self,x):
                    return self.model(x).mean(0)
            model1 = model_1(model)
            model1.to(device)
            atk = AutoAttack(model1, norm='L1', eps=args.eps,version='custom',attacks_to_run=['apgd-ce'],verbose=False)
        else:
            print(f'Clean, {args.encoding} model encoding')
            atk = None
        

        #acc = val(model, test_loader, device, args.time, atk)
        acc = val2(model, test_loader, device, args.time, args.m, atk)

        # logger.info(json.dumps(atk_config)+' Test acc={:.3f}'.format(acc))
        print('final Test Accu: ', acc)
        logger.info(f'Final Test Accu: {acc}')

if __name__ == "__main__":
    main()
