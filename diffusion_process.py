import numpy as np
import torch
import pdb

class CSDI_Generation():

    def __init__(self, model, config):
        self.diffmodel = model
        self.timesteps = config["num_steps"]

        if config["schedule"] == "quad":
            self.beta = np.linspace(
                config["beta_start"] ** 0.5, config["beta_end"] ** 0.5, self.timesteps, dtype=np.float32
            ) ** 2
        elif config["schedule"] == "linear":
            self.beta = np.linspace(
                config["beta_start"], config["beta_end"], self.timesteps, dtype=np.float32
            )

        self.beta = torch.from_numpy(self.beta)


    def forward_diffusion(self, batch):

        B, K, L = batch.shape
        t = torch.randint(0, self.timesteps, (B,), device=batch.device)
        noise = torch.randn_like(batch, device=batch.device)
        alpha = torch.sqrt(1 - self.beta)
        alpha_cumprod = torch.cumprod(alpha, 0)

        xt = (alpha_cumprod[t] ** 0.5).unsqueeze(-1).unsqueeze(-1) * batch + torch.sqrt(1 - alpha_cumprod[t]).unsqueeze(-1).unsqueeze(-1) * noise

        return xt, t, noise

    def reverse_diffusion(self, x):
        for t in reversed(range(self.timesteps)):
            noise_pred = self.diffmodel(x, t)
            x = (x - self.beta[t] * noise_pred) / torch.sqrt(1 - self.beta[t])
        return x



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