import torch
import torch.nn as nn
from torchattacks.attack import Attack


class SEA(Attack):
    r"""
    altered from torchattack
    """
    def __init__(self, model, device, forward_function=None, eps=0.007, T=None, signed=False, **kwargs):
        super().__init__("FGSM", model)
        
        self.eps = int(eps)
        self._supported_mode = ['default', 'targeted']
        self.forward_function = forward_function
        self.T = T
        self.signed = signed

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        if self.forward_function is not None:
            outputs = self.forward_function(self.model, images, self.T)
        else:
            outputs = self.model(images)    

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        
        if self.signed:
            grad_sign = torch.sign(grad)
            delta_hat = grad_sign - images * torch.abs(grad_sign)
        else:
            grad_sign = torch.sign(grad) # 1, 0, -1
            g_ind  = 1*(grad < 0)        # 0, 1   
            delta_hat = grad_sign*((1-g_ind)*(1-images)+ g_ind * images)
        
        #print(delta_hat[0])
        
        B = grad.shape[0]
        _, topk_indices = torch.topk((delta_hat * grad).view(B, -1), k=self.eps, dim=1)
        delta = torch.zeros_like(grad)
        delta.view(B, -1)[torch.arange(B).unsqueeze(1), topk_indices] = delta_hat.view(B, -1)[torch.arange(B).unsqueeze(1), topk_indices]
        adv_images = images + delta
        return adv_images #with temporal dimension