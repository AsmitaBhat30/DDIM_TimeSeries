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




    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1 - gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp