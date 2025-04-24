import logging
import os
import shutil
from typing import Optional

import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plantseg.training.model import UNet2D
from plantseg.training.utils import RunningAverage

logger = logging.getLogger(__name__)


# adapted from https://github.com/wolny/pytorch-3dunet
class UNetTrainer:
    """UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): learning rate scheduler
        loss_criterion (nn.Module): loss function
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_iterations (int): maximum number of iterations
        device (str): device to use for training
        log_after_iters (int): number of iterations before logging to tensorboard
        pre_trained(str): path to the pre-trained model
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.ReduceLROnPlateau,
        loss_criterion: nn.Module,
        loaders: dict,
        checkpoint_dir: str,
        max_num_iterations: int,
        device: str = "cuda",
        log_after_iters: int = 100,
        pre_trained: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_iterations = max_num_iterations
        self.max_num_epochs = max_num_iterations // len(loaders["train"]) + 1
        self.device = device
        self.log_after_iters = log_after_iters
        self.best_eval_loss = float("+inf")

        self.num_iterations = 1
        if pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            state = torch.load(pre_trained, map_location="cpu")
            self.model.load_state_dict(state)

        # init tensorboard logger
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "logs"))

    def train(self) -> None:
        for epoch in range(self.max_num_epochs):
            logger.info(f"Epoch [{epoch}/~{self.max_num_epochs}]")
            # train for one epoch
            should_terminate = self.train_epoch()

            if should_terminate:
                logger.info("Stopping criterion is satisfied. Finishing training")
                return

            logger.info("Validating...")
            # set the model in eval mode
            self.model.eval()
            # evaluate on validation set
            eval_loss = self.validate()
            logger.info(f"Val Loss: {eval_loss}.")
            self.writer.add_scalar("Loss/val", eval_loss, self.num_iterations)
            # set the model back to training mode
            self.model.train()

            # adjust learning rate if necessary
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
            # remember best validation metric
            is_best = eval_loss < self.best_eval_loss
            if is_best:
                self.best_eval_loss = eval_loss

            # save checkpoint
            self._save_checkpoint(is_best)

        logger.info(
            f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training..."
        )

    def train_epoch(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()

        # sets the model in training mode
        self.model.train()

        for _input, target in tqdm(self.loaders["train"]):
            _input, target = _input.to(self.device), target.to(self.device)
            output, loss = self._forward_pass(_input, target)

            train_losses.update(loss.item(), self._batch_size(_input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.log_after_iters == 0:
                # log stats, params and images
                logger.info(f"Train Loss: {train_losses.avg}.")
                self.writer.add_scalar(
                    "Loss/train", train_losses.avg, self.num_iterations
                )

                if self.should_stop():
                    return True

            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(
                f"Maximum number of iterations {self.max_num_iterations} exceeded."
            )
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]["lr"]
        if lr < min_lr:
            logger.info(f"Learning rate below the minimum {min_lr}.")
            return True

        return False

    def validate(self):
        val_losses = RunningAverage()

        with torch.no_grad():
            for _input, target in tqdm(self.loaders["val"]):
                _input, target = _input.to(self.device), target.to(self.device)

                output, loss = self._forward_pass(_input, target)
                val_losses.update(loss.item(), self._batch_size(_input))

            return val_losses.avg

    def _forward_pass(
        self, _input: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.model, UNet2D):
            # remove the singleton z-dimension from the input
            _input = torch.squeeze(_input, dim=-3)
            # forward pass
            output = self.model(_input)
            # add the singleton z-dimension to the output
            output = torch.unsqueeze(output, dim=-3)
        else:
            # forward pass
            output = self.model(_input)

        loss = self.loss_criterion(output, target)
        return output, loss

    def _save_checkpoint(self, is_best):
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, "last_checkpoint.pytorch")
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        torch.save(state_dict, last_file_path)
        if is_best:
            logger.info("Saving best checkpoint")
            best_file_path = os.path.join(
                self.checkpoint_dir, "best_checkpoint.pytorch"
            )
            shutil.copyfile(last_file_path, best_file_path)

    @staticmethod
    def _batch_size(_input):
        if isinstance(_input, (list, tuple)):
            return _input[0].size(0)
        else:
            return _input.size(0)
