# coding:utf-8

import torch
from torch import nn

class VisEmb(nn.Module):
    
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(VisEmb, self).__init__()
        
        emb_dim = cfg.vis_emb_dims # 64

        embedding_tower = []
    
        for index in range(cfg.vis_num_convs):

            if index==0:
                in_channels_new = in_channels + 2
            else:
                in_channels_new = in_channels

            embedding_tower.append(
                nn.Conv2d(
                    in_channels_new,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                ))
            embedding_tower.append(nn.GroupNorm(32, in_channels))
            embedding_tower.append(nn.ReLU())


        self.add_module('embedding_tower', nn.Sequential(*embedding_tower))
            
        self.visual_embed = nn.Conv2d(
            in_channels, emb_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):

        visual_embedding = self.visual_embed(self.embedding_tower(x))

        return visual_embedding