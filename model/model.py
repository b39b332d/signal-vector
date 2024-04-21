import torch 
import torch.nn as nn
import sys,os
sys.path.append(os.getcwd())
from base import BaseModel
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SigNet(BaseModel): 
    def __init__(self, dropout=0.0):
        super().__init__()
        self.layer1 = nn.Sequential(
           nn.ZeroPad2d((15,15,0,0)),
           nn.Conv2d(in_channels = 3, out_channels = 20, kernel_size = (1,31), stride = (1,1), padding = 0),
           nn.LeakyReLU(),
           #nn.Dropout(p=dropout)
           )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 20, out_channels = 40, kernel_size = (1,2), stride = (1,2), padding = 0),
            nn.BatchNorm2d(40, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = (1,3), stride = (1,2))
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels = 80, kernel_size = (1,21), stride = (1,1)),
            nn.LeakyReLU(),
            #nn.Dropout(p=dropout)
            )
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))
        self.layer4 = nn.Sequential(
            #nn.ZeroPad2d((15,15,0,0)),
            nn.Conv2d(in_channels=80, out_channels = 160, kernel_size = (1,11), stride = (1,1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))
        
        
        self.layer5a = nn.Sequential(
            nn.Conv2d(in_channels = 160, out_channels = 160, kernel_size = (1,7), stride=(1,1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU())
        self.layer5b = nn.Sequential(
            nn.Conv2d(in_channels = 160, out_channels = 160, kernel_size = (1,7), stride=(1,1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU())
        self.pool4a = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))
        self.pool4b = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))
        self.linear1a = nn.Linear(160*2, 1)
        self.linear1b = nn.Linear(160*2, 1)
        
            
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out= self.layer3(out)
        out = self.pool2(out)
        out = self.layer4(out)
        out = self.pool3(out)
        outa = self.layer5a(out)
        outb = self.layer5b(out)
        outa = self.pool4a(outa)
        outb = self.pool4b(outb)
        outa = torch.flatten(outa,start_dim=1)
        outb = torch.flatten(outb,start_dim=1)
        outa= self.linear1a(outa)
        outb= self.linear1b(outb)
        return outa,outb
class SigNet2(BaseModel): 
    def __init__(self, dropout=0.0):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels = 52, out_channels = 256, kernel_size = 31, stride = 1, padding = "same"),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2)
           #nn.Dropout(p=dropout)
           )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size = 11, stride = 1, padding = 0),
            nn.BatchNorm1d(512, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 4, stride = 4)
            )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels = 1024, kernel_size = 7, stride = 2, padding = 0),
            nn.BatchNorm1d(1024, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 3)
            #nn.Dropout(p=dropout)
            )
        
        self.layer4a = nn.Sequential(
            nn.Conv1d(in_channels = 1024, out_channels = 2048, kernel_size = 3, stride=1),
            nn.BatchNorm1d(2048, affine=False),
            nn.LeakyReLU())
        self.layer5a = nn.Linear(4096, 52)
        
            
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out= self.layer3(out)
        outa = self.layer4a(out)
        outa = torch.flatten(outa,start_dim=1)
        outa = outa[:, None, :]
        outa = self.layer5a(outa)
        return torch.matmul(outa,x)
    
    def loss(self,output, target):
        o=torch.flatten(output)
        t=torch.flatten(target)
        loss = 1-torch.corrcoef(torch.vstack((o,t)))[0,1]
        return loss

class SigNet3(BaseModel): 
    def __init__(self, dropout=0.0):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels = 52, out_channels = 256, kernel_size = 31, stride = 1, padding = "same"),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2)
           #nn.Dropout(p=dropout)
           )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size = 11, stride = 1, padding = 0),
            nn.BatchNorm1d(512, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 4, stride = 4)
            )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels = 1024, kernel_size = 7, stride = 2, padding = 0),
            nn.BatchNorm1d(1024, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 3)
            #nn.Dropout(p=dropout)
            )
        
        self.layer4a = nn.Sequential(
            nn.Conv1d(in_channels = 1024, out_channels = 2048, kernel_size = 3, stride=1),
            nn.BatchNorm1d(2048, affine=False),
            nn.LeakyReLU())
        self.layer5a = nn.Linear(4096, 52)
        self.loss = nn.L1Loss()
        
            
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out= self.layer3(out)
        outa = self.layer4a(out)
        outa = torch.flatten(outa,start_dim=1)
        outa = outa[:, None, :]
        outa = self.layer5a(outa)
        return outa
    
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0.0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding

    params:
        kernel_size: kernel size
        stride: the stride of the window. Default value is kernel_size
    
    input: (n_sample, n_channel, n_length)
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        p = max(0, self.kernel_size - 1)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0.0)
        
        net = self.max_pool(net)
        
        return net
    
class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)

class BasicBlock(nn.Module):
    """
    Basic Block: 
        conv1 -> convk -> conv1

    params:
        in_channels: number of input channels
        out_channels: number of output channels
        ratio: ratio of channels to out_channels
        kernel_size: kernel window length
        stride: kernel step size
        groups: number of groups in convk
        downsample: whether downsample length
        use_bn: whether use batch_norm
        use_do: whether use dropout

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups, downsample, is_first_block=False, use_bn=True, use_do=True):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.middle_channels = int(self.out_channels * self.ratio)

        # the first conv, conv1
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=self.in_channels, 
            out_channels=self.middle_channels, 
            kernel_size=1, 
            stride=1,
            groups=1)

        # the second conv, convk
        self.bn2 = nn.BatchNorm1d(self.middle_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=self.middle_channels, 
            out_channels=self.middle_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the third conv, conv1
        self.bn3 = nn.BatchNorm1d(self.middle_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(
            in_channels=self.middle_channels, 
            out_channels=self.out_channels, 
            kernel_size=1, 
            stride=1,
            groups=1)

        # Squeeze-and-Excitation
        r = 2
        self.se_fc1 = nn.Linear(self.out_channels, self.out_channels//r)
        self.se_fc2 = nn.Linear(self.out_channels//r, self.out_channels)
        self.se_activation = Swish()

        if self.downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        out = x
        # the first conv, conv1
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv, convk
        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # the third conv, conv1
        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out) # (n_sample, n_channel, n_length)

        # Squeeze-and-Excitation
        se = out.mean(-1) # (n_sample, n_channel)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = F.sigmoid(se) # (n_sample, n_channel)
        out = torch.einsum('abc,ab->abc', out, se)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0.0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out

class BasicStage(nn.Module):
    """
    Basic Stage:
        block_1 -> block_2 -> ... -> block_M
    """
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups, i_stage, m_blocks, use_bn=True, use_do=True, verbose=False):
        super(BasicStage, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.i_stage = i_stage
        self.m_blocks = m_blocks
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        self.block_list = nn.ModuleList()
        for i_block in range(self.m_blocks):
            
            # first block
            if self.i_stage == 0 and i_block == 0:
                self.is_first_block = True
            else:
                self.is_first_block = False
            # downsample, stride, input
            if i_block == 0:
                self.downsample = True
                self.stride = stride
                self.tmp_in_channels = self.in_channels
            else:
                self.downsample = False
                self.stride = 1
                self.tmp_in_channels = self.out_channels
            
            # build block
            tmp_block = BasicBlock(
                in_channels=self.tmp_in_channels, 
                out_channels=self.out_channels, 
                ratio=self.ratio, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=self.groups, 
                downsample=self.downsample, 
                is_first_block=self.is_first_block,
                use_bn=self.use_bn, 
                use_do=self.use_do)
            self.block_list.append(tmp_block)

    def forward(self, x):

        out = x

        for i_block in range(self.m_blocks):
            net = self.block_list[i_block]
            out = net(out)
            if self.verbose:
                print('stage: {}, block: {}, in_channels: {}, out_channels: {}, outshape: {}'.format(self.i_stage, i_block, net.in_channels, net.out_channels, list(out.shape)))
                print('stage: {}, block: {}, conv1: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv1.in_channels, net.conv1.out_channels, net.conv1.kernel_size, net.conv1.stride, net.conv1.groups))
                print('stage: {}, block: {}, convk: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv2.in_channels, net.conv2.out_channels, net.conv2.kernel_size, net.conv2.stride, net.conv2.groups))
                print('stage: {}, block: {}, conv1: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv3.in_channels, net.conv3.out_channels, net.conv3.kernel_size, net.conv3.stride, net.conv3.groups))

        return out

class Net1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    params:
        in_channels
        base_filters
        filter_list: list, filters for each stage
        m_blocks_list: list, number of blocks of each stage
        kernel_size
        stride
        groups_width
        n_stages
        n_classes
        use_bn
        use_do

    """

    def __init__(self, in_channels, base_filters, ratio, filter_list, m_blocks_list, kernel_size, stride, groups_width, n_classes, use_bn=True, use_do=True, verbose=False):
        super(Net1D, self).__init__()
        
        #self.loss = nn.L1Loss()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.ratio = ratio
        self.filter_list = filter_list
        self.m_blocks_list = m_blocks_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups_width = groups_width
        self.n_stages = len(filter_list)
        self.n_classes = n_classes
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        # first conv
        self.first_conv = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=self.base_filters, 
            kernel_size=self.kernel_size, 
            stride=2)
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()

        # stages
        self.stage_list = nn.ModuleList()
        in_channels = self.base_filters
        for i_stage in range(self.n_stages):

            out_channels = self.filter_list[i_stage]
            m_blocks = self.m_blocks_list[i_stage]
            tmp_stage = BasicStage(
                in_channels=in_channels, 
                out_channels=out_channels, 
                ratio=self.ratio, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=out_channels//self.groups_width, 
                i_stage=i_stage,
                m_blocks=m_blocks, 
                use_bn=self.use_bn, 
                use_do=self.use_do, 
                verbose=self.verbose)
            self.stage_list.append(tmp_stage)
            in_channels = out_channels

        # final prediction
        self.dense = nn.Linear(in_channels, n_classes)
        
    def forward(self, x):
        
        out = x
        # out = torch.view_as_real(torch.fft.rfft(x)[:,:,1:]).flatten(2,3)
        # first conv
        out = self.first_conv(out)
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)
        
        # stages
        for i_stage in range(self.n_stages):
            net = self.stage_list[i_stage]
            out = net(out)

        # final prediction
        out = out.mean(-1)
        out = self.dense(out)
        # return out
        return torch.matmul(out[:,None,:],x)
    
    def loss(self,output, target):
        o=torch.flatten(output)
        t=torch.flatten(target)
        loss = 1-torch.corrcoef(torch.vstack((o,t)))[0,1]
        return loss
    
# define the model using pytorch
class ConvNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(36, 64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Linear(1600,200),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(200,36),
            #nn.Softmax())
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # return out
        return torch.matmul(out[:,None,:],x)
    def loss(self,output, target):
        o=torch.flatten(output)
        t=torch.flatten(target)
        loss = 1-torch.corrcoef(torch.vstack((o,t)))[0,1]
        return loss

def default_net():
    return ConvNet1D()
    # return Net1D(
    #     in_channels=24,
    #     base_filters=100,
    #     ratio=1.0,
    #     filter_list = [64, 128, 128, 256,256,512],
    #     m_blocks_list = [4, 4, 6, 6,8,8],
    #     kernel_size=16,
    #     stride=2,
    #     groups_width=16,
    #     verbose=False,
    #     n_classes=24)



def load_model(model_savepath=None):
    device = torch.device('cuda:0')
    model = None
    if model_savepath is not None and model_savepath[-3:] == ".pt":
        model = torch.jit.load(model_savepath)
    else:
        model = default_net()
        if model_savepath is not None:
            checkpoint = torch.load(model_savepath)
            model.load_state_dict(checkpoint)

    model.to(device)
    torch.set_default_device(device)
    return model

def save_model(model=None,model_checkpoint=None,out_path="model_scripted.pt"):
    X=torch.tensor(np.zeros([1,36,256]), dtype=torch.float)
    if model is None:
        model = default_net()
    if model_checkpoint is not None:
        checkpoint = torch.load("model_save/model_20240419_120154_179_0.37475821375846863")
        model.load_state_dict(checkpoint)
    _=model(X)
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(out_path) # Save

if __name__ == "__main__":
    #net = SigNet3()
    n_block = 8
    base_filters = 64
    net = Net1D(
        in_channels=52,
        base_filters=100,
        ratio=1.0,
        filter_list = [128, 384, 384, 1024, 1024],
        m_blocks_list = [2, 2, 3, 3, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        n_classes=52)

    import numpy as np
    out=torch.tensor(np.zeros([1,256]), dtype=torch.float)
    X=torch.tensor(np.zeros([1,52,256]), dtype=torch.float)
    y=net(X)
    l = net.loss(y,out)
    torch.onnx.export(net, X, "model.onnx")
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter("torchlogs/")
    writer.add_graph(net, X)
    writer.close()
