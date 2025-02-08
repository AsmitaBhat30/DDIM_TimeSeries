import numpy as np
import torch
import pdb

class CSDI_Generation():

    def __init__(self, model, config, device):
        self.diffmodel = model
        self.timesteps = config["num_steps"]
        self.device = device

        if config["schedule"] == "quad":
            self.beta = np.linspace(
                config["beta_start"] ** 0.5, config["beta_end"] ** 0.5, self.timesteps, dtype=np.float32
            ) ** 2
        elif config["schedule"] == "linear":
            self.beta = np.linspace(
                config["beta_start"], config["beta_end"], self.timesteps, dtype=np.float32
            )

        self.beta = torch.from_numpy(self.beta).to(self.device)


    def forward_diffusion(self, batch):

        B, K, L = batch.shape
        t = torch.randint(0, self.timesteps, (B,)).to(self.device)
        noise = torch.randn_like(batch).to(self.device)
        #pdb.set_trace()
        alpha = torch.sqrt(1 - self.beta).to(self.device)
        alpha_cumprod = torch.cumprod(alpha, 0).to(self.device)

        xt = (alpha_cumprod[t] ** 0.5).unsqueeze(-1).unsqueeze(-1) * batch + torch.sqrt(1 - alpha_cumprod[t]).unsqueeze(-1).unsqueeze(-1) * noise

        return xt, t, noise
