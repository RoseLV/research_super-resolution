import copy

import blobfile as bf
import lightning as L
import numpy as np
import torch

from improved_diffusion.train_util import get_blob_logdir, log_loss_dict

from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class DDPM(L.LightningModule):
    def __init__(
        self,
        model,
        diffusion,
        lr,
        ema_rate,
        schedule_sampler=None,
        weight_decay=0.0,
    ):
        super(DDPM, self).__init__()
        self.model = model
        self.diffusion = diffusion
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay

        self.model_params = list(self.model.parameters())
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.ema_params = [
            copy.deepcopy(self.model_params) for _ in range(len(self.ema_rate))
        ]

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.5, 0.9),
            weight_decay=self.weight_decay,
        )
        step_lr = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[300, 500], gamma=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": step_lr,
        }

    def log_loss_dict(self, diffusion, ts, losses):
        for key, values in losses.items():
            self.log(key, values.mean().item())
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                self.log(f"{key}_q{quartile}", sub_loss)

    def shared_step(self, batch, batch_idx):
        hr, lr = batch["hr"], batch["lr"]
        t, weights = self.schedule_sampler.sample(hr.shape[0], "cuda")

        losses = self.diffusion.training_losses(
            self.model, hr, t, model_kwargs={"low_res": lr}
        )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_all_losses(t, losses["loss"].detach())
        loss = (losses["loss"] * weights).mean()
        log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()
        self.manual_backward(loss)

        sqsum = 0.0
        for p in self.model_params:
            sqsum += (p.grad**2).sum().item()
        self.log("grad_norm", np.sqrt(sqsum))

        opt.step()
        sch.step()

        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model_params, rate=rate)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint):
        self.save()

    def get_blob_logdir(self):
        return "TODO"

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if not rate:
                filename = f"model{(self.global_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.global_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.global_step):06d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)
