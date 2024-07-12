import copy

import blobfile as bf
import lightning as L
import numpy as np
import torch

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
        model_dir="checkpoints",
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
        self.model_dir = model_dir
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.5, 0.9),
            weight_decay=self.weight_decay,
        )
        step_lr = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 300], gamma=0.1
        )
        cos_lr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": cos_lr,
        }

    def log_loss_dict(self, diffusion, ts, losses, stage: str):
        for key, values in losses.items():
            self.log(f"{stage}_{key}", values.mean().item(), prog_bar=True)
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                self.log(f"{stage}_{key}_q{quartile}", sub_loss)

    def shared_step(self, batch, batch_idx, stage: str):
        hr, lr = batch["hr"], batch["lr"]
        t, weights = self.schedule_sampler.sample(hr.shape[0], "cuda")

        losses = self.diffusion.training_losses(
            self.model, hr, t, model_kwargs={"low_res": lr}
        )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_all_losses(t, losses["loss"].detach())
        loss = (losses["loss"] * weights).mean()
        self.log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}, stage
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "train")

        opt = self.optimizers()

        opt.zero_grad()
        self.manual_backward(loss)

        sqsum = 0.0
        for p in self.model_params:
            sqsum += (p.grad**2).sum().item()
        self.log("grad_norm", np.sqrt(sqsum))

        opt.step()

        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step()
            self.log("lr", sch.get_last_lr()[0])

        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model_params, rate=rate)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def on_save_checkpoint(self, checkpoint):
        self.save()

    def get_blob_logdir(self):
        return self.model_dir

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if not rate:
                filename = f"model{(self.global_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.global_step):06d}.pt"
            with bf.BlobFile(bf.join(self.get_blob_logdir(), filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(0, self.model_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

    def _master_params_to_state_dict(self, model_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = model_params[i]
        return state_dict
