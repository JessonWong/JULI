# Copyright 2025-2026 The JULI Authors
# SPDX-License-Identifier: Apache-2.0

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch
import torch.nn as nn
import os
class BiasConfig(PretrainedConfig):
    model_type = "biaas_net"
    
    def __init__(
        self,
        hidden_size=None,
        vocab_size=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

class BiasNet(PreTrainedModel):
    config_class = BiasConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.intermediate_size = self.hidden_size // 2
        self.vocab_size = config.vocab_size
        self.layer1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.layer2 = nn.Linear(self.intermediate_size, self.intermediate_size)
        self.final_projection = nn.Linear(self.intermediate_size, self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)       
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(0.1)
    
    def set_up_proj(self):
        if self.layer1.weight.dtype in [torch.float]:
            self.up_proj = torch.linalg.pinv(self.lm_head.weight.clone().detach().t())
        elif self.layer1.weight.dtype in [torch.half]:
            self.up_proj = torch.linalg.pinv(self.lm_head.weight.float().clone().detach().t()).half()
            print("half")
    def inverse_mapping(self, logits):
        if len(logits.shape) == 3: #[bsz, seq_len, vocab_size]
            hidden_states = torch.matmul(logits[-1], self.up_proj)
        elif len(logits.shape) == 2: #[bsz, vocab_size]
            hidden_states = torch.matmul(logits, self.up_proj)
        return hidden_states
    def forward(self, logits):
        x = self.inverse_mapping(logits)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.final_projection(x)
        logits = self.lm_head(x)
        return logits
    def save_pretrained(self, save_dir, **kwargs):
        os.makedirs(save_dir, exist_ok=True)
        self.config.save_pretrained(save_dir)
        model_state = {
            'bias_network': self.state_dict(),
            'config': self.config
        }
        model_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(model_state, model_path)
    @classmethod
    def from_pretrained(cls, save_dir, map_location="cpu"):
        config = BiasConfig.from_pretrained(save_dir)
        model = cls(config)
        model_path = os.path.join(save_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            model_state = torch.load(model_path, map_location=map_location, weights_only=False)
            model.load_state_dict(model_state['bias_network'])
        return model
