# coding:utf-8

import os
import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(MAIN_PATH))

import torch
import torch.nn as nn

from transformers import T5ForConditionalGeneration

class TransformerProgramGenerator(nn.Module):
    """This class is for generating solution program according to the parse results."""
    
    def __init__(self, cfg, save_dir):
        super(TransformerProgramGenerator, self).__init__()
        
        self.cfg = cfg
        
        # self.t5_model = T5ForConditionalGeneration.from_pretrained(cfg.model_type)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join(save_dir, "t5/t5"))
        
    def forward(self, input_ids, attention_mask, target_ids=None):
        
        if self.training:
            output = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
            
            losses = {"program_loss": output.loss}
            
            return losses

        else:
            
            output = self.t5_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=self.cfg.beam_size,
                num_return_sequences=self.cfg.beam_size,
                max_length=300,
            )
            
            return output