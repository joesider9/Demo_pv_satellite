# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 19:44:25 2024

@author: ra064640
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttUnet(nn.Module):
    def __init__(self, segmentation, f_int, time_step, layer=4, channel_1=64, channel_2=128,
                 channel_3=256, channel_4=512, channel_5=1024):
        super(AttUnet, self).__init__()
        self.layer = layer
        self.seg = segmentation
        self.f_int = f_int
        self.time_step = time_step
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.channel_3 = channel_3
        self.channel_4 = channel_4
        self.channel_5 = channel_5
        self.channels_list = [3, self.channel_1, self.channel_2, self.channel_3,
                              self.channel_4, self.channel_5]
        self.relu = nn.ReLU
        self.sigmoid = nn.sigmoid()
        self.conv_maxpool = nn.MaxPool2d(kernel_size=2)

        # create convolutin layers.
        self.conv_front_layers = []
        self.conv_back_layers = []

        for i in range(self.layer + 1):
            self.conv_front_layers.append(nn.Conv2d(in_channels=self.channels_list[i],
                                                    out_channels=self.channels_list[i + 1],
                                                    kernel_size=3, stride=1, padding=1))
            self.conv_back_layers.append(nn.Conv2d(in_channels=self.channels_list[i + 1],
                                                   out_channels=self.channels_list[i + 1],
                                                   kernel_size=3, stride=1, padding=1))

        # create the deconvolution layers.
        self.deconv_up_layers = []
        self.deconv_conv_front = []
        self.deconv_conv_back = []

        for i in range(self.layers):
            # i need to check the size of the output after each of the deconvolution
            self.deconv_up_layers.append(nn.ConvTranspose2d(in_channels=self.channels_list[i + 1],
                                                            out_channels=self.channels_list[i],
                                                            kernel_size=2,
                                                            ))
            # the front one needs to take the concatenated one and reduce feature by half
            self.deconv_conv_front.append(nn.Conv2d(in_channels=self.channels_list[i + 1],
                                                    out_channels=self.channels_list[i],
                                                    kernel_size=3, padding=1))
            # second convolutin keeps the feature map size
            self.deconv_conv_back.append(nn.Conv2d(in_channels=self.channels_list[i],
                                                   out_channels=self.channels_list[i],
                                                   kernel_siz=3, padding=1))
        # last 1 x 1 convolution
        self.last_deconv_conv = nn.Conv2d(in_channels=self.channel_1,
                                          out_channels=self.seg, kernel_size=1)

        # set up the attention layers.
        self.x_down_layer = []
        self.x_layer = []
        self.g_layer = []

        for i in range(self.layers):
            # x layer needs to be down- sampled to g. need to check dimensions if they work.
            self.x_down_layer.append(nn.Conv2d(in_channels=self.channels_list[(self.layers) - i],
                                               out_channels=self.channels_list[(self.layers) - i],
                                               kernel_size=2, stride=2))

            # carry out Wx 1x1x1 convolution - just have to change the channel size
            self.x_layer.append(nn.Conv2d(in_channels=self.channels_list[self.layers - i],
                                          out_channels=self.f_int,
                                          kernel_size=1, stride=1))

            # carry out the Wg 1x1x1 convoution - just have to change the channel size
            self.g_layer.append(nn.Conv2d(in_channels=self.channels_list[(self.layers + 1) - i],
                                          out_channels=self.f_int,
                                          kernel_size=1, stride=1))

            # phi layer to one
        self.phi = nn.Conv2d(in_channels=self.f_int,
                             out_channels=1,
                             kernel_size=1, stride=1)

    def forward(self, x, t):
        # embed the time step to  x by adding
        x = x + t
        # convolution layer output: save the output after the second cnn
        conv_output = []
        for i, (cnn1, cnn2) in enumerate(zip(self.conv_front_layers, self.conv_back_layers)):
            # first cnn1
            cnn1_out = cnn1(x)
            # second cnn2
            cnn2_out = cnn2(cnn1_out)
            conv_output.append(cnn2_out)
            # only do max pool until self.layer: there are 5 cnn, but no max pool after the last cnn
            # since self.layer = 4 and "i" will go up to 4, i < self.layer
            if i < self.layer:
                # max pool
                max_out = self.conv_maxpool(conv_output)
                x = max_out
            else:
                x = cnn2_out
        # start deconvolution, and attention

        for i, (deconv, x_down, Wx_conv, Wg_conv, sig, deconv_cnn1, deconv_cnn2) \
                in enumerate(zip(self.deconv_up_layers, self.x_down_layer, self.x_layer, self.g_layer,
                                 self.sigmoid, self.deconv_conv_front, self.deconv_conv_back)):
            # do deconv
            deconv_out = deconv(x)
            ################ attention start ######################
            # x layer - this is from the cnn2 (conv_output)
            x_l = conv_output[self.layers - (i + 1)]
            x_l_downsample = x_down(x_l)  # reduce the size to g.
            Wx_out = Wx_conv(x_l_downsample)
            # g layer - this is just a concolution needed
            g = conv_output[(self.layers) - i]
            Wg_out = Wg_conv(g)
            # add then relu
            WxWg_out = Wx_out + Wg_out
            relu_out = self.relu(WxWg_out)
            # phi layer + sigmoid
            phi_out = self.phi(relu_out)
            sig_out = sig(phi_out)
            # resample + element mulitplication
            output_tensor = F.interpolate(
                sig_out,
                scale_factor=(1, 2, 2),  # Scale depth by 1 (unchanged), H and W by 2
                mode='trilinear',  # Use trilinear interpolation
                align_corners=False  # Recommended for smooth interpolation
            )
            att_out = x_l * output_tensor.unsqueeze(0)
            ############### finish attention ######################
            # concatenate attention out and the deconvolution
            cat_tensor = torch.cat((att_out, deconv_out), dim=0)
            # first cnn for deconv
            deconv_cnn1_out = deconv_cnn1(cat_tensor)
            # second cnn for deconv
            deconv_cnn2_out = deconv_cnn2(deconv_cnn1_out)

            # loop it back (assign the output as input of the loop)
            x = deconv_cnn2_out
        ### last convolution output
        final_out = self.last_deconv_conv(x)
        return final_out