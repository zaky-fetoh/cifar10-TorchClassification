import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, drop_out=.5,
                 force_booling=True, add_skip_conn=False):
        super().__init__()
        for arg in 'in_ch,out_ch,drop_out,kernel_size,force_booling,add_skip_conn'.split(','):
            exec('self.' + arg + '=' + arg)
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=self.padding)
        self.bn = nn.BatchNorm2d(out_ch)
        if self.drop_out:
            self.do = nn.Dropout2d(p=drop_out)
        else:
            self.do = None
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.do is not None:
            out = self.do(out)
        out = f.relu(out)
        if self.add_skip_conn:
            out = torch.cat([x, out], dim=-3)
            if self.force_booling:
                return f.max_pool2d(out, 2)
            else:
                return out
        else:
            return f.max_pool2d(out, 2)


class Dense(nn.Module):
    def __init__(self, input, layers_sizes: list):
        super().__init__();
        self.ml = nn.ModuleList()
        for ls in layers_sizes:
            ly = nn.Linear(input, ls)
            nn.init.xavier_normal_(ly.weight)
            self.ml.append(ly)
            input = ls

    def forward(self, x):
        for lyr in self.ml[:-1]:
            x = f.relu(lyr(x))
        return self.ml[-1](x)


def get_seq_fatten_size(input_shape, conv_layers_params):
    convLayers = nn.ModuleList();
    input = input_shape[0]
    for out, ks, do, fb, sk in conv_layers_params:
        convLayers.append(ConvBlock(input, out, ks, do, fb, sk))
        if sk:
            input += out
        else:
            input = out
    seq = nn.Sequential(*convLayers)
    bs = torch.rand((1,) + input_shape)
    return np.prod([x for x in seq(bs).shape])


class ConvModel(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), conv_layers_params=[(256, 5, .6, True, True),
                                                                    (128, 3, .5, False, True),
                                                                    (64, 3, .4, True, False),
                                                                    (32, 3, .4, True, True),
                                                                    (16, 3, .2, False, False)],
                 linear_layers_size=[64, 10, 10], device='auto'):
        super().__init__();
        input = input_shape[0]
        convLayers = nn.ModuleList()
        for out, ks, do, fb, sk in conv_layers_params:
            convLayers.append(ConvBlock(input, out, ks, do, fb, sk))
            if sk:
                input += out
            else:
                input = out
        self.ConvBase = nn.Sequential(*convLayers)
        self.fltt_inp = get_seq_fatten_size(input_shape, conv_layers_params)
        self.DenseBase = Dense(self.fltt_inp, linear_layers_size)

        if device == 'auto':
            self.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.to(device=device)

    def forward(self, x):
        internalState = self.ConvBase(x)
        internalState = internalState.view(-1, self.fltt_inp)
        return self.DenseBase(internalState)


def saveModel(model: nn.Module, filename=r'.\auxfile\model.pt'):
    sd = model.state_dict()
    torch.save(sd, filename)


def loadModel(filename=r'.\auxfile\model.pt'):
    sd = torch.load(filename)
    model = ConvModel()
    model.load_state_dict(sd)
    return model


if __name__ == '__main__':
    layer = ConvBlock(3, 5, add_skip_conn=True, force_booling=True)
    img = torch.rand((2, 3, 32, 32)).to(device='cuda')
    cmodel = ConvModel()
    out = cmodel(img)
