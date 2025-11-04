import torch
from typing import Optional
import torch.nn as nn
import wandb

def reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(x,-1)
    return x - cumsum + cumsum[..., -1:None]

class PiecewiseTemperatureFunction:
    @staticmethod
    def apply_simple_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        return logits * temperature.clamp(min=1e-4)
        
    @staticmethod
    def bin_by_rank(logits: torch.Tensor, # (B, V)
                    counts: torch.Tensor, # (P-1,) if summing up to V, (P,)
                    ) -> torch.Tensor:
        sorted_logits, idx = torch.sort(logits, dim=-1)
        total_count = counts.sum()
        if total_count < logits.size(-1):
            counts = torch.cat([
                (logits.size(-1) - total_count).unsqueeze(0),
                counts
                ], dim=-1)
        bin_idx = torch.arange(
            counts.size(-1), device=logits.device
            ).repeat_interleave(counts).unsqueeze(0).expand(logits.shape) # (B, V)
        bin_ids = logits.new_zeros(logits.shape, dtype=torch.int)
        bin_ids[idx] = bin_idx
        cut_ids = torch.cumsum(counts, dim=0)
        thresholds = sorted_logits[:,cut_ids[:-1]] # (B, P-1)
        return bin_ids, thresholds
    
    @staticmethod
    def relative_shifts(logits: torch.Tensor, # (B,V)
                        ratios: torch.Tensor, # (P,)
                        ) -> torch.Tensor: # (B, V)
        """Get bins by shifts relative to max logit. Ratios will be normalized to sum to 1.
         E.g., ratios = [0.5, 0.3, 0.2] means thresholds are min, max - (max-min)*0.5, max - (max-min)*0.2, max """
        P = ratios.size(0)
        B = logits.size(0)
        min_vals = logits.min(dim=-1).values.unsqueeze(-1).expand(B, P-1) # (B, P)
        max_vals = logits.max(dim=-1).values.unsqueeze(-1).expand(B, P-1) # (B, P)
        ratios = (ratios / ratios.sum(dim=-1))[1:].unsqueeze(0).expand(B, P-1) # Make sure ratios add up to one
        return torch.cumsum(ratios, dim=-1) * (max_vals-min_vals) + min_vals # (B, P-1)
    
    @staticmethod
    def absolute_shifts(logits: torch.Tensor, # (B,V)
                        shifts: torch.Tensor, # (P-1,)
                        ) -> torch.Tensor: # (B, V)
        """Get bins by relative shifts from max logit.
         E.g., shifts = [10, 3, 1] means thresholds are max-14, max-4, max-1"""
        shifts = reverse_cumsum(shifts) # (P-1,)
        return logits.max(dim=-1).values.unsqueeze(-1) - shifts.unsqueeze(0) # (B, P-1)

    @staticmethod  
    def get_constants_no_grad(temperatures: torch.Tensor,  # (P,)
                    thresholds: torch.Tensor,    # (P-1,) or (B, P-1)
                            ):
        temp_diff = - torch.diff(temperatures, dim=-1) # (a1-a2, a2-a3, ....) : (P-1, )
        c = torch.cumsum(temp_diff * thresholds, dim=-1) # (P-1,) or (B, P-1)
        return torch.nn.functional.pad(c, (1,0), value=0.0) # (P,) or (B, P)
    
    @staticmethod
    def apply_temperature(
                    logits: torch.Tensor, # (B, V)
                    temperatures: torch.Tensor, # (P,)
                    thresholds: torch.Tensor, # (P-1,) or (B, P-1)
                    bin_ids: torch.Tensor, # (B, V)
                    use_no_grad_version: bool = False,
                    ) -> torch.Tensor: # (B, V)
        require_grad = temperatures.requires_grad or thresholds.requires_grad or logits.requires_grad
        if require_grad and not use_no_grad_version:
            return PiecewiseTemperatureFunction.apply_temperature_with_grad(
                logits, temperatures, thresholds, bin_ids
            )
        else:
            return PiecewiseTemperatureFunction.apply_temperature_no_grad(
                logits, temperatures, thresholds, bin_ids
            )
        
    @staticmethod
    def apply_temperature_no_grad(
                logits: torch.Tensor, # (B, V)
                temperatures: torch.Tensor, # (P,)
                thresholds: torch.Tensor, # (P-1,) or (B, P-1)
                bin_ids: torch.Tensor, # (B, V)
                ) -> torch.Tensor: # (B, V)
        c = PiecewiseTemperatureFunction.get_constants_no_grad(temperatures, thresholds) # (P,) or (B, P)
        return logits * temperatures[bin_ids] + c[bin_ids]

    @staticmethod  
    def get_constants_with_grad(temperatures: torch.Tensor,  # (P,)
                                thresholds: torch.Tensor,    # (P-1,) or (B, P-1)
                                ):
        differences = torch.diff(torch.nn.functional.pad(thresholds, (1, 0), value=0.0), dim=-1)
        return torch.cumsum(differences * temperatures[:-1], dim=-1)
    
    @staticmethod
    def apply_temperature_with_grad(
                logits: torch.Tensor, # (B, V)
                temperatures: torch.Tensor, # (P,)
                thresholds: torch.Tensor, # (P-1,) or (B, P-1)
                bin_ids: torch.Tensor, # (B, V)
                ) -> torch.Tensor: # (B, V)
        P = temperatures.size(0)
        B, V = logits.shape
        c = PiecewiseTemperatureFunction.get_constants_with_grad(temperatures, thresholds) # (P-1,) or (B, P-1)
        logits = logits.unsqueeze(2).expand(B,V,P) # (B, V, P)
        thresholds = (thresholds.unsqueeze(0).unsqueeze(0)# (P-1,) -> (B,V,P)
                      if thresholds.dim() == 1
                      else thresholds.unsqueeze(1) # (B, P-1) -> (B,V,P)
                      ).expand(B, V, P-1)

        new_logits = logits.new_zeros(logits.shape)
        c = (c.unsqueeze(0).unsqueeze(0)# (P-1,) -> (B,V,P)
            if c.dim() == 1
            else c.unsqueeze(1) # (B, P-1) -> (B,V,P)
            ).expand(B, V, P-1)
        new_logits[:,:,0] = logits[:,:,0] * temperatures[0]
        new_logits[:,:,1:] = temperatures[1:] * (logits[:,:,1:] - thresholds) + c
        return new_logits.gather(2, bin_ids.unsqueeze(2)).squeeze(2)

class TemperatureScaler(nn.Module):
    """Trainable temperature scaling module."""
    def __init__(self,
                mode: str = "absolute", # "rank", "absolute", "relative_shifts", "absolute_shifts", "none" for single piece
                n_temp: int = 1,
                pieces: int = 1,
                thresholds: Optional[torch.Tensor] = None, # (pieces - 1,)
                counts: Optional[torch.Tensor] = None, # (pieces - 1,) or (pieces,) if summing up to vocab size
                ratios: Optional[torch.Tensor] = None, # (pieces,)
                shifts: Optional[torch.Tensor] = None, # (pieces-1,)
                init_temp: str = "ones", # "ones" or "random",
                init_thresholds: str = "linspace", # (pieces - 1,)
                threshold_min: float = -30.0,
                threshold_max: float = 30.0,
                temp_mask_threshold: Optional[float] = None,
                temp_mask: bool = False,
                ):
        super().__init__()
        self.mode = mode
        self.n_temp = n_temp
        self.pieces = pieces
        self.temp_mask_threshold = temp_mask_threshold
        self.temp_mask = temp_mask
        self.use_no_grad_version = False
        if counts is not None:
            self.register_buffer("counts", counts)  
        else:
            self.counts = None
        # Temperature parameters
        if init_temp == "ones":
            self.temperature = nn.Parameter(torch.ones(n_temp, pieces, dtype=torch.float32))
        elif init_temp == "random":
            self.temperature = nn.Parameter(torch.rand(n_temp, pieces, dtype=torch.float32))
        if pieces > 1:
            
            if mode == "absolute":
                if init_thresholds == "linspace":
                    self.thresholds = nn.Parameter(
                        torch.linspace(threshold_min, threshold_max, self.pieces-1)
                        .unsqueeze(0).repeat(self.n_temp, 1)
                    )
                elif init_thresholds == "random":
                    self.thresholds = nn.Parameter(
                        (threshold_max - threshold_min) * torch.rand(self.n_temp, self.pieces - 1) + threshold_min
                    )
                else:
                    raise ValueError(f"Unknown init_thresholds: {init_thresholds}")
                if thresholds is not None:
                    with torch.no_grad():
                        self.thresholds.copy_(thresholds.unsqueeze(0).repeat(self.n_temp, 1))

            elif mode == "absolute_shifts":
                self.shifts = nn.Parameter(
                    torch.zeros(self.n_temp, self.pieces - 1, dtype=torch.float32)
                )
                if shifts is not None:
                    with torch.no_grad():
                        self.shifts.copy_(shifts.unsqueeze(0).repeat(self.n_temp, 1))
            elif mode == "relative_shifts":
                self.ratios = nn.Parameter(
                    torch.ones(self.n_temp, self.pieces, dtype=torch.float32) / pieces
                )
                if ratios is not None:
                    with torch.no_grad():
                        self.ratios.copy_(ratios.unsqueeze(0).repeat(self.n_temp, 1))
            else:
                self.register_parameter('thresholds', None)

    def mask_temp_and_threshold(self, thresholds, temps):
        if self.temp_mask_threshold is None or not self.temp_mask:
            return thresholds, temps
        temp_diff = torch.diff(temps, dim=-1).abs()
        temp_mask = temp_diff > self.temp_mask_threshold
        if temp_mask.sum() == 0: #Return at least one threshold and two temps
            temp_mask[-1] = True  
        return thresholds[temp_mask], temps[nn.functional.pad(temp_mask, (1,0), value=True)]

    def forward(self,
                logits: torch.Tensor,
                i: int = 0,
                return_thresholds: bool = False) -> torch.Tensor:
        """Apply temperature scaling to logits.
        
        Args:
            logits: (B, V) logits tensor
            i: Index of temperature set to use
            
        Returns:
            Temperature-scaled logits
        """
        if self.pieces == 1:
            return logits * self.temperature[i, 0].clamp(min=1e-4)
        else:
            bin_ids = None
            if self.mode == "absolute":
                thresholds = torch.sort(self.thresholds[i]).values
                temperatures = self.temperature[i].clamp(min=1e-4)
                masked_thresholds, masked_temperatures = self.mask_temp_and_threshold(thresholds, temperatures)
                bin_ids = torch.searchsorted(masked_thresholds, logits)
                logits = PiecewiseTemperatureFunction.apply_temperature(
                    logits, masked_temperatures, masked_thresholds, bin_ids,
                    use_no_grad_version=getattr(self, "use_no_grad_version",False)
                )
            else:
                temperatures = self.temperature[i].clamp(min=1e-4)
                if self.mode == "rank":
                    bin_ids, thresholds = PiecewiseTemperatureFunction.bin_by_rank(
                        logits, self.counts
                    )
                elif self.mode == "absolute_shifts":
                    thresholds = PiecewiseTemperatureFunction.absolute_shifts(
                        logits, self.shifts[i]
                    )
                elif self.mode == "relative_shifts":
                    thresholds = PiecewiseTemperatureFunction.relative_shifts(
                        logits, self.ratios[i]
                    )
                bin_ids = torch.searchsorted(thresholds, logits)
                logits = PiecewiseTemperatureFunction.apply_temperature(
                    logits, self.temperature[i], thresholds, bin_ids,
                    use_no_grad_version=getattr(self, "use_no_grad_version",False)
                )
            if return_thresholds:
                return logits, thresholds
            else:
                return logits
        
    @torch.no_grad()   
    def get_logs(self,):
        out = {}
        for i in range(self.n_temp):
            if self.pieces == 1:
                continue
            xmin, xmax = -30, 30 # Default range for plotting
            if getattr(self, "thresholds", None) is not None:
                thresholds = self.thresholds[i].detach()
                xmin = thresholds.min() - 5
                xmax = thresholds.max() + 5
            xs = torch.linspace(xmin, xmax, 500).to(self.temperature.device)
            y, thresholds = self.forward(xs.unsqueeze(0), i, return_thresholds=True)
            y = y.squeeze()
            thresholds = thresholds.squeeze(0) if thresholds.dim() == 2 else thresholds
            y = y - y[-1] + xs[-1] # Shift so that f(x_max) = x_max
            # Log thresholds
            out.update({
                    f"group_{i}_threshold_{j}" : threshold
                     for j, threshold in enumerate(thresholds.cpu().tolist())
                })
            
            for name in ["shifts", "ratios", "temperature"]:
                if getattr(self, name, None) is not None:
                    values = getattr(self, name)[i].detach()
                    out.update({
                        f"group_{i}_{name}_{j}" : value
                        for j, value in enumerate(values.cpu().tolist())
                    })

            xs = xs.squeeze().cpu().tolist()
            y = y.squeeze().cpu().tolist()
            out[f"group_{i}_plot"] = wandb.plot.line_series(
                        xs=xs,
                        ys=[y, xs],
                        keys=["f(x)", "y=x"],
                        title=f"Temperature {i} Plot",
                        xname="x"
                    )
        return out
    