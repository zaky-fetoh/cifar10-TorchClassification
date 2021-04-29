import torch.optim as opt
import model as modelfile
import data_org as dtorg
import torch.nn as nn
import utility as u
import torch

tr, va = dtorg.get_loaders()
model = modelfile.ConvModel(device='cpu')
optimizer = opt.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
u.train(model, tr, loss_fn, optimizer,device='cpu')


