from cv2 import mean
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F
import numpy as np
from models.layers import rate_encode, const_encode


def arsnn_reg(net, beta):
    l = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            weight = m.weight
            if isinstance(m, nn.Conv2d):
                weight = weight.view(weight.shape[0], -1)
            sum_1 = torch.sum(F.relu(0 - weight), dim=1)
            sum_2 = torch.sum(F.relu(weight), dim=1)
            l += (torch.max(sum_1) + torch.max(sum_2)) * beta
    return l

def TET_loss(outputs, labels, criterion, means=1, lamb=1e-3,T = 8):
    Loss_es = 0
    # outputs = expand(outputs)
    # print(outputs.shape,labels.shape)
    for t in range(T):
        Loss_es += criterion(outputs[t, ...], labels)
    Loss_es = Loss_es / T # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y) # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd # L_Total

def train(model, device, train_loader, criterion, optimizer, T, atk, beta, parseval=False,TET=False):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for i, (images, labels) in enumerate((train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        #encoding = model.encoding
        if atk.__class__.__name__=='SEA':
            atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            images = rate_encode(images, T, model.signed)
            images = atk(images, labels.to(device)) # with T
        elif atk is not None: #gn, fgsm, pgd
            atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            atk.model.model_encode=True               
            images = atk(images, labels.to(device)) # without T
            atk.model.model_encode = False
            images = rate_encode(images, T, model.signed)  #with T
        else:
            images = rate_encode(images, T, model.signed)  #with T
        
        if T > 0:
            outs = model(images)
            outputs = outs.mean(0)   
        else:
            outputs = model(images)

        if TET:
            loss = TET_loss(outs, labels, criterion,T=T)
        else:
            loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        if parseval:
            orthogonal_retraction(model, beta)
            convex_constraint(model)
    
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

def val(model, test_loader, device, T, atk=None):
    correct = 0
    total = 0
    model.eval()
    #encoding = model.encoding
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)
   
        if atk.__class__.__name__=='SEA':
            atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            #atk.model.model_encode=False
            inputs = rate_encode(inputs, T, model.signed)
            inputs = atk(inputs, targets.to(device))  #with T
        elif atk is not None:
            atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            atk.model.model_encode=True
            inputs = atk(inputs, targets.to(device)) #without T
            atk.model.model_encode=False
            inputs = rate_encode(inputs, T, model.signed) #with T
        else:
            inputs = rate_encode(inputs, T, model.signed) #with T

        with torch.no_grad():
            if T > 0:
                outputs = model(inputs).mean(0)
            else:
                outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    final_acc = 100 * correct / total
    print(total)
    return final_acc

def smoothed(model, inputs, m):
# rate-encodes the input for m times and returns the output
    model.model_encode=True
    with torch.no_grad():
        outputs = model(inputs).mean(0)
        for i in range(m-1):
            outputs += model(inputs).mean(0)

    _, predicted = outputs.cpu().max(1)
    return predicted


def val2(model, test_loader, device, T, m, atk=None):
    correct = 0
    total = 0
    model.eval()
    #encoding = model.encoding
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)
   
        if atk.__class__.__name__=='SEA':
            atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            atk.model.model_encode=False
            inputs = rate_encode(inputs, T, model.signed)
            inputs = atk(inputs, targets.to(device))  #with T
        elif atk is not None:
            atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            atk.model.model_encode=True
            inputs = atk(inputs, targets.to(device)) #without T
            predicted = smoothed(model, inputs, m)
        else:
            predicted = smoothed(model, inputs, m)   

        # with torch.no_grad():
        #     if T > 0:
        #         outputs = model(inputs).mean(0)
        #     else:
        #         outputs = model(inputs)

        # print(inputs.shape, outputs.shape)                
        # _, predicted = outputs.cpu().max(1)
            

        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    final_acc = 100 * correct / total
    print(total)
    return final_acc



def ip(a,b):
    return torch.sum(a*b, dim=(1,2,3))

def help_signed_k(x, delta):
    x_abs = torch.abs(x)
    x_plus = x*torch.ge(x,0)
    x_minus = -x*(x<0)
    adv_plus = (x+delta)*torch.ge(x+delta,0)
    adv_minus = - (x+delta)*(x+delta<0)
    return ip(x_abs,torch.ones_like(x))+ip(1-x_abs, torch.abs(x+delta))-ip(x_plus,adv_plus)-ip(x_minus,adv_minus) - 2*ip(x_abs, 1-x_abs)

def compute_k(model, data_loader, device, T, atk=None):
    k_list = []
    model.eval()
    #encoding = model.encoding
    for batch_idx, (inputs, targets) in enumerate((data_loader)):
        inputs = inputs.to(device)
   
        if atk.__class__.__name__=='SEA':
            print("Attack not supported.")
        elif atk is not None:
            atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            atk.model.model_encode=True
            adv_inputs = atk(inputs, targets.to(device)) #without T
            atk.model.model_encode=False
            delta = adv_inputs - inputs
            if not model.signed:
                k = torch.sum(delta*(1-2*inputs), dim=(1,2,3))  # implementing eqn.(20) - eqn.(9) 
            else:
                k = help_signed_k(inputs, delta)  # implementing eqn.(21) - eqn.(12)
            k_list.extend(k.cpu().numpy())

        else:
            if model.signed:    
                inputs = torch.abs(inputs)
            k = 2*torch.sum(inputs*(1-inputs), dim=(1,2,3)) 
            k_list.extend(k.cpu().numpy())

        
    return k_list



def getSmooth(model, inputs, m):
    hold_out = torch.empty(size=(inputs.shape[0],m), dtype=int)
    for i in range(m):
        outputs = model(inputs).mean(0)
        hold_out[:,i] = outputs.max(1)[1]

    temp=torch.empty_like(outputs, device='cpu')
    for i, a in enumerate(hold_out):
        temp[i] = torch.bincount(a, minlength=outputs.shape[1])

    return temp

# def certify(model, test_loader, device, T, atk=None, m0=10, m=20, error_rate=None, radius=0.1):
#     correct = 0
#     total_predicted = 0
#     abstain = 0
#     model.eval()
#     print(f'T={T}, m0={m0}, m={m}, error_rate={error_rate}, radius={radius}')
#     for batch_idx, (inputs, targets) in enumerate((test_loader)):
#         inputs = inputs.to(device)
#         temp1 = getSmooth(model, inputs, m0)    
#         _, predicted = temp1.max(1)

#         temp2 = getSmooth(model, inputs, m)    
#         counts, _ = temp2.max(1)

#         if error_rate!=None:
#             p_a = torch.empty(size=(targets.shape[0],), device='cpu', dtype=float)
#             for i, c in enumerate(counts):
#                 p_a[i] = proportion_confint( count=int(c.item()), nobs=m, alpha=2*error_rate, method="beta")[0]
            
#             l1_rad = p_a -0.5
#         else:
#             prob = temp2.topk(k=2)[0]/m
#             l1_rad = (prob[:,0] - prob[:,1])/2
            
#         idx = torch.where(~(l1_rad < radius))[0] # Good Case
#         abstain += targets.size(0) - len(idx)
      
#         total_predicted += float(targets[idx].size(0))
#         correct += float(predicted[idx].eq(targets[idx]).sum().item())

#     final_acc = 100 * correct / (total_predicted+abstain)
#     abs_per = 100* abstain /(total_predicted+abstain)

#     return final_acc, abs_per


# def predict(model, test_loader, device, T, atk=None, m=1, error_rate=None):
#     correct = 0
#     total = 0
#     abstain = 0
#     total_predicted = 0
#     model.eval()
#     print(f'T={T}, m={m}')
#     for batch_idx, (inputs, targets) in enumerate((test_loader)):
#         inputs = inputs.to(device)

#         if atk is not None:
#             atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
#             atk.model.model_mode='attack'
#             inputs = atk(inputs, targets.to(device))
#             model.model_mode = 'normal'
            
#         temp = getSmooth(model,inputs, m)
#         _, predicted = temp.max(1)
        
#         if error_rate != None:
#             counts = temp.topk(k=2)[0]
#             p_value = torch.empty(size=(targets.shape[0],), device='cpu', dtype=float)
#             for i, c in enumerate(counts):
#                 p_value[i] = binomtest( k=int(c[0].item()), n=int((c[0]+c[1]).item()), p=0.5).pvalue
#                 # print(p_value[i])
#             idx = torch.where(p_value <= error_rate)[0]
            
#             abstain += targets.size(0) - len(idx) 
            
#         total_predicted += float(targets[idx].size(0))
#         correct += float(predicted[idx].eq(targets[idx]).sum().item())

#     final_acc = 100 * correct / (total_predicted+abstain)
#     abs_per = 100* abstain /(abstain+total_predicted)

#     return final_acc, abs_per


def orthogonal_retraction(model, beta=0.002):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(module, nn.Conv2d):
                    weight_ = module.weight.data
                    sz = weight_.shape
                    weight_ = weight_.reshape(sz[0],-1)
                    rows = list(range(module.weight.data.shape[0]))
                elif isinstance(module, nn.Linear):
                    if module.weight.data.shape[0] < 200: # set a sample threshold for row number
                        weight_ = module.weight.data
                        sz = weight_.shape
                        weight_ = weight_.reshape(sz[0], -1)
                        rows = list(range(module.weight.data.shape[0]))
                    else:
                        rand_rows = np.random.permutation(module.weight.data.shape[0])
                        rows = rand_rows[: int(module.weight.data.shape[0] * 0.3)]
                        weight_ = module.weight.data[rows,:]
                        sz = weight_.shape
                module.weight.data[rows,:] = ((1 + beta) * weight_ - beta * weight_.matmul(weight_.t()).matmul(weight_)).reshape(sz)


def convex_constraint(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ConvexCombination):
                comb = module.comb.data
                alpha = torch.sort(comb, descending=True)[0]
                k = 1
                for j in range(1,module.n+1):
                    if (1 + j * alpha[j-1]) > torch.sum(alpha[:j]):
                        k = j
                    else:
                        break
                gamma = (torch.sum(alpha[:k]) - 1)/k
                module.comb.data -= gamma
                torch.relu_(module.comb.data)