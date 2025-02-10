import argparse
import os
from utils import compute_k
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-data', '--dataset', default='cifar10',type=str,help='dataset')
parser.add_argument('-id1', '--identifier1', type=str, help='model statedict identifier')
parser.add_argument('-id2', '--identifier2', type=str, help='model statedict identifier')
parser.add_argument('--attack',default='',type=str,help='attack')
parser.add_argument('-eps','--eps',default=8,type=float,metavar='N',help='attack eps')

args = parser.parse_args()


def main():
    global args
    log_dir = '%s-Results'% (args.dataset)
    log_dir = os.path.join(log_dir, 'stats')
    
    if len(args.attack) == 0:
        f_name =os.path.join(log_dir, str(args.identifier1)+'_list.pkl')
        with open(f_name, 'rb') as handle:
            k1= np.array(pickle.load(handle))
        
        f_name =os.path.join(log_dir, str(args.identifier2)+'_list.pkl')
        with open(f_name, 'rb') as handle:
            k2= np.array(pickle.load(handle))

        print(np.mean(k1), np.mean(k2), np.mean(k1)/np.mean(k2), np.mean(k1/k2))
    else:
        f_name = f'{args.identifier1}_attack_{args.attack}_epsilon_{args.eps}_list.pkl'
        f_name = os.path.join(log_dir,f_name)
        with open(f_name, 'rb') as handle:
            k1= np.array(pickle.load(handle))
        print(f'mean={np.mean(k1):.2f}, pos_mean = {np.mean((k1>0)*k1):.2f}, abs_mean={np.mean(abs(k1)):.2f}, min={np.min(k1):.2f}, max={np.max(k1):.2f}, shape={k1.shape}')


if __name__ == "__main__":
    main()
