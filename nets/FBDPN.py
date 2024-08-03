from collections import OrderedDict

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from model.ResNet50_4out import ResNet50

from einops import rearrange
from nets.Swin_transformer import SwinTransformerBlock

#   1*1 Conv with ReLU
class Conv_1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
#   1*1 Conv without ReLU
class Conv_1x1_noReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
        )
    def forward(self, x):
        x = self.conv(x)
        return x

#   Conv、BN、ReLU
def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#   Conv、BN
def conv2d_noreLu(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
    ]))

#   NSFBM
class NSFBM(nn.Module):
    def __init__(self, channels, resolutions:tuple, resolution):
        super(NSFBM,self).__init__()
        self.channels = channels
        self.resolution = resolution
        self.STB1 = SwinTransformerBlock(channels, resolutions, 4)
        self.STB2 = SwinTransformerBlock(channels, resolutions, 4)
        self.conv1x1_attention = Conv_1x1(channels * 2, channels)
        self.conv1x1_out = conv2d(channels * 2, channels, 1)
        self.fc_attention = nn.Linear(channels, channels)

    def Interaction_Branch(self, Fa_reshape, Fb_reshape):
        fu1_tf = self.STB1(Fa_reshape)
        fu2_tf = self.STB2(Fb_reshape)

        fu1_tf_reshape = rearrange(fu1_tf, 'b (h w) c -> b h w c', h=self.resolution, w=self.resolution)
        fu2_tf_reshape = rearrange(fu2_tf, 'b (h w) c -> b h w c', h=self.resolution, w=self.resolution)

        fu1_fu2 = (torch.cat((fu1_tf_reshape, fu2_tf_reshape), dim=3)).permute(0, 3, 1, 2)
        fu1_fu2_reshape = self.conv1x1_attention(fu1_fu2)

        return fu1_fu2_reshape
    
    def correlation(self, Fa, Fb):
        co_out = Fa * Fb

        return co_out
        
    def Correlation_Branch(self, x):
        GAP_x = torch.mean(torch.mean(x, dim=2, keepdim= True), dim=3, keepdim= True)
        fc = torch.relu(self.fc_attention(GAP_x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        Dual_attention = x * fc

        return Dual_attention
        
    def forward(self, Fa, Fb):
        Fa_reshape = rearrange(Fa, 'b c h w -> b (h w) c')  # b (H W) C
        Fb_reshape = rearrange(Fb, 'b c h w -> b (h w) c')

        # deal with Interaction Branch and Correlation Branch
        Interaction_Branch_out = self.Interaction_Branch(Fa_reshape, Fb_reshape)
        correlation_out = self.correlation(Fa, Fb)
        attention_out = self.Correlation_Branch(correlation_out)

        # Channel concatenation
        out_cat = torch.cat((Interaction_Branch_out, attention_out), dim= 1)
        out = self.conv1x1_out(out_cat)

        return out

#   CSFDM SCC
class CNN1(nn.Module):
    def __init__(self, channel, map_size, pad):
        super(CNN1,self).__init__()
        self.weight = nn.Parameter(torch.ones(channel,channel,map_size,map_size),requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(channel),requires_grad=False)
        self.pad = pad
        self.norm = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = F.conv2d(x, self.weight, self.bias, stride=1, padding=self.pad)
        out = self.norm(out)
        out = self.relu(out)
        return out
    
#   CSFDM
class CSFDM(nn.Module):
    def __init__(self, CSFDM_out_Channel:int, conv_3, conv_5):
        super().__init__()
        self.conv_3 = conv_3    # 3*3 Conv
        self.conv_5 = conv_5    # 5*5 Conv
        self.out_deal = nn.Sequential(
            nn.Conv2d(CSFDM_out_Channel, CSFDM_out_Channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(CSFDM_out_Channel), 
            nn.LeakyReLU(inplace=True))
        
    def forward(self, Bi, Bj, size_Bi:list):
        Bj_up = F.interpolate(Bj, size= size_Bi, mode= 'bilinear')
        Bi_map1 = self.conv_3(Bi)
        Bj_up_map1 = self.conv_3(Bj_up)
        Bi_map2 = self.conv_5(Bi)
        Bj_up_map2 = self.conv_5(Bj_up)
        out = self.out_deal(abs(Bj_up - Bi) + abs(Bj_up_map1 - Bi_map1) + abs(Bj_up_map2 - Bi_map2))
        return out

#   **************************************FBDPN******************************************    #
class FBDPN(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(FBDPN, self).__init__()

        self.last_layers_number0 = len(anchors_mask[0]) * (num_classes + 5)
        self.last_layers_number1 = len(anchors_mask[1]) * (num_classes + 5)
        self.last_layers_number2 = len(anchors_mask[2]) * (num_classes + 5)
     
        self.Bb_Conv0 = conv2d(256, 64, 3)  
        self.Bb_Conv1 = conv2d(512, 64, 3)  
        self.Bb_Conv2 = conv2d(1024, 64, 3)
        self.Bb_Conv3 = conv2d(2048, 64, 3)

        # NSFBM
        self.NSFBM1  = NSFBM(64, (80, 80), 80)
        self.NSFBM2  = NSFBM(64, (40, 40), 40)
        self.NSFBM3  = NSFBM(64, (20, 20), 20)

        # CSFDM
        self.conv_3_1 = CNN1(64, 3, 1)    
        self.conv_5_1 = CNN1(64, 5, 2)

        self.conv_3_2 = CNN1(64, 3, 1)    
        self.conv_5_2 = CNN1(64, 5, 2)

        self.conv_3_3 = CNN1(64, 3, 1)    
        self.conv_5_3 = CNN1(64, 5, 2)

        self.CSFDM1 = CSFDM(64, self.conv_3_1, self.conv_5_1)   
        self.CSFDM2 = CSFDM(64, self.conv_3_2, self.conv_5_2)
        self.CSFDM3 = CSFDM(64, self.conv_3_3, self.conv_5_3)

        # DFCM
        self.DFCM_weights1 = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0]), requires_grad=True)
        self.DFCM_weights2 = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 2.0]), requires_grad=True)
        self.DFCM_weights3 = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=True)

        # FNDPN Out
        self.FBDPN_Conv1 = conv2d(64, 64, 3)  
        self.FBDPN_Conv2 = conv2d(64, 64, 3)
        self.FBDPN_Conv3 = conv2d(64, 64, 3)
 
        # Head
        self.predict1 = Conv_1x1_noReLU(64, self.last_layers_number0)   
        self.predict2 = Conv_1x1_noReLU(64, self.last_layers_number1)
        self.predict3 = Conv_1x1_noReLU(64, self.last_layers_number2)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels        
                m.weight.data.normal_(0, math.sqrt(2. / n))                     
            elif isinstance(m, nn.BatchNorm2d):                                 
                m.weight.data.fill_(1)                                          
                m.bias.data.zero_()                                             

        # Backbone
        self.Backbone = ResNet50()

    def forward(self, x):
        input = x   # bs, 3, 640, 640

        # ----------Backbone-----------------
        Backbone_out = self.Backbone(x)

        Backbone_out0 = self.Bb_Conv0(Backbone_out[0]) #   bs, 64, 160, 160 -> bs, 64, 160, 160
        Backbone_out1 = self.Bb_Conv1(Backbone_out[1]) #   bs, 512, 80, 80 -> bs, 64, 80, 80
        Backbone_out2 = self.Bb_Conv2(Backbone_out[2]) #   bs, 1024, 40, 40 -> bs, 64, 40, 40
        Backbone_out3 = self.Bb_Conv3(Backbone_out[3]) #   bs, 2048, 20, 20 -> bs, 64, 20, 20
        # -----------------------------------

        # ----------Backbone interpolate-----------
        Backbone_out0_1 = F.interpolate(Backbone_out0, size= [80, 80], mode= 'bilinear')    # bs, 64, 80, 80
        Backbone_out1_2 = F.interpolate(Backbone_out1, size= [40, 40], mode= 'bilinear')    # bs, 64, 40, 40
        Backbone_out1_3 = F.interpolate(Backbone_out1, size= [20, 20], mode= 'bilinear')    # bs, 64, 20, 20
        Backbone_out2_1 = F.interpolate(Backbone_out2, size= [80, 80], mode= 'bilinear')    # bs, 64, 80, 80
        Backbone_out2_3 = F.interpolate(Backbone_out2, size= [20, 20], mode= 'bilinear')    # bs, 64, 20, 20
        Backbone_out3_1 = F.interpolate(Backbone_out3, size= [80, 80], mode= 'bilinear')    # bs, 64, 80, 80
        Backbone_out3_2 = F.interpolate(Backbone_out3, size= [40, 40], mode= 'bilinear')    # bs, 64, 40, 40

        # --------NSFBM-------------
        NSFBM_1 = self.NSFBM1(Backbone_out0_1, Backbone_out1) + Backbone_out1   # bs, 64, 80, 80
        NSFBM_2 = self.NSFBM2(Backbone_out1_2, Backbone_out2) + Backbone_out2   # bs, 64, 40, 40
        NSFBM_3 = self.NSFBM3(Backbone_out2_3, Backbone_out3) + Backbone_out3   # bs, 64, 20, 20


        # ----------CSFDM--------------
        CSFDM_B1_B2 = self.CSFDM1(NSFBM_1, NSFBM_2, [80, 80])
        CSFDM_B1_B3 = self.CSFDM2(NSFBM_1, NSFBM_3, [80, 80])
        CSFDM_B2_B3 = self.CSFDM3(NSFBM_2, NSFBM_3, [40, 40])
        # ------------------------------------

        # -----------DFCM---------------
        DFCM_weights1_sigmoid = torch.sigmoid(self.DFCM_weights1)
        DFCM_weights2_sigmoid = torch.sigmoid(self.DFCM_weights2)
        DFCM_weights3_sigmoid = torch.sigmoid(self.DFCM_weights3)
        #-------------------------------------

        # -----------FBDPN Out------------------
        FBDPN_out1 = self.FBDPN_Conv1(Backbone_out1 * DFCM_weights1_sigmoid[0] + Backbone_out2_1 * DFCM_weights1_sigmoid[1] + Backbone_out3_1 * DFCM_weights1_sigmoid[2] + \
                                  CSFDM_B1_B2 * DFCM_weights1_sigmoid[3] + CSFDM_B1_B3 * DFCM_weights1_sigmoid[4])
        FBDPN_out2 = self.FBDPN_Conv2(Backbone_out2 * DFCM_weights2_sigmoid[0] + Backbone_out1_2 * DFCM_weights2_sigmoid[1] + Backbone_out3_2 * DFCM_weights2_sigmoid[2] + \
                                  CSFDM_B2_B3 * DFCM_weights2_sigmoid[3])
        FBDPN_out3 = self.FBDPN_Conv3(Backbone_out3 * DFCM_weights3_sigmoid[0] + Backbone_out1_3 * DFCM_weights3_sigmoid[1] + Backbone_out2_3 * DFCM_weights3_sigmoid[2])
        # ------------------------------------

        # ------------Out-------------------
        out0 = self.predict1(FBDPN_out1)
        out1 = self.predict2(FBDPN_out2)
        out2 = self.predict3(FBDPN_out3)
        # ----------------------------------

        return out2, out1, out0
