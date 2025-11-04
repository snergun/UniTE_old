import torch
import torch.nn as nn

class ModelWithTemperature(nn.Module):
    def __init__(self, base_model, temperature_scaler):
        super().__init__()
        self.base_model = base_model
        self.temperature_scaler = temperature_scaler
    
    def generate(self, **kwargs):
        outputs = self.base_model.generate(**kwargs)
        # Apply temperature scaling to logits
        if hasattr(outputs, 'logits'):
            outputs.logits = self.temperature_scaler(outputs.logits)
        return outputs