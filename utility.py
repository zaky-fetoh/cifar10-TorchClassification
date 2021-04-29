import torchvision.utils as vutils
import matplotlib.pyplot as plt
import model as modelfile
from time import time
import torch


def train(model, tr_loader, loss_fn, optimizer, epochs=30, device='auto'):
    d = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu')
    dicloss = dict()
    for ep in range(epochs):
        tik = time()
        for imgs, lbls in tr_loader:
            imgs = imgs.to(d);  dicloss[ep] = list()
            lbls = lbls.to(d)
            pred = model(imgs)
            loss = loss_fn(pred, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # memoing
            dicloss[ep].append(loss.item())
        tok = time();
        modelfile.saveModel(model, 'cmodel%3i.pt' % (ep))
        avl = sum(dicloss[ep]) / len(tr_loader)/128
        print("epoch %03i take %0.6f ms loss is %0.3f" % (ep, (tok - tik), avl))

def visualize_model_kernels(model):
    firstlayer = model.ConvBase[0].conv.weight.detach()
    kg = vutils.make_grid(firstlayer,nrow=16,padding=1,scale_each=True)
    kg = kg.permute(1, 2, 0).to(device='cpu').detach().numpy()
    return kg

def get_val_acc(model, val_loadr):
    pass


if __name__ == '__main__':
    model = modelfile.loadModel('model  0.pt')
    kg,lh = visualize_model_kernels(model)

