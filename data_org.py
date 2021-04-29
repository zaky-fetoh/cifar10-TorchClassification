import torchvision.transforms as transforms
import torchvision.datasets as vdatasets
import torch.utils.data as data
import torch

ROOT = '\cifar10'

Scale = lambda x: (x - x.min()) / (x.max() - x.min())
Noise = lambda x: x + torch.rand_like(x).normal_( ) * 0.3



img_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(30)])

tr_img = vdatasets.CIFAR10(ROOT, download=True,
                           transform=transforms.Compose([
                               img_aug, transforms.ToTensor(),
                               transforms.Lambda(Scale),
                               transforms.Lambda(Noise)]))

va_img = vdatasets.CIFAR10(ROOT, train= False,download= True,
                           transform = transforms.ToTensor())

def get_dts_loader(dts, batch_size=128, device='auto'):
    pin_memory = True  if device == 'auto' and torch.cuda.is_available() else False
    tr_loader = data.DataLoader(dts,batch_size, shuffle=True,
                                pin_memory=pin_memory)
    return tr_loader

def get_loaders(tr=tr_img, va=va_img, constr= get_dts_loader):
    return constr(tr), constr(va)

