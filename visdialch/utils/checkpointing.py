"""
A checkpoint manager periodically saves model and optimizer as .pth
files during training.

Checkpoint managers help with experiment reproducibility, they record
the commit SHA of your current codebase in the checkpoint saving
directory. While loading any checkpoint from other commit, they raise a
friendly warning, a signal to inspect commit diffs for potential bugs.
Moreover, they copy experiment hyper-parameters as a YAML config in
this directory.

That said, always run your experiments after committing your changes,
this doesn't account for untracked or staged, but uncommitted changes.
"""
from pathlib import Path

import torch

from torch import nn, optim

from anatool import AnaLogger


class CheckpointManager:
    """A checkpoint manager saves state dicts of model and optimizer
    as .pth files in a specified directory. This class closely follows
    the API of PyTorch optimizers and learning rate schedulers.

    Note::
        For ``DataParallel`` modules, ``model.module.state_dict()`` is
        saved, instead of ``model.state_dict()``.

    Parameters
    ----------
    model: nn.Module
        Wrapped model, which needs to be checkpointed.
    optimizer: optim.Optimizer
        Wrapped optimizer which needs to be checkpointed.
    checkpoint_dirpath: str
        Path to an empty or non-existent directory to save checkpoints.
    step_size: int, optional (default=1)
        Period of saving checkpoints.
    last_epoch: int, optional (default=-1)
        The index of last epoch.
    """

    def __init__(
            self,
            model,
            optimizer,
            logger: AnaLogger,
            checkpoint_dirpath,
            step_size=1,
            last_epoch=-1,
    ):

        if not isinstance(model, nn.Module):
            logger.error('{} is not a Module'.format(type(model).__name__))
            raise TypeError

        if not isinstance(optimizer, optim.Optimizer):
            logger.error('{} is not an Optimizer'.format(type(optimizer).__name__))
            raise TypeError

        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.ckpt_dirpath = Path(checkpoint_dirpath)
        self.step_size = step_size
        self.last_epoch = last_epoch
        self.init_directory()

    def init_directory(self):
        """Initialize empty checkpoint directory."""
        self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)

    def step(self, epoch=None):
        """Save checkpoint if step size conditions meet. """

        if not epoch:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # if not self.last_epoch % self.step_size:
        #     torch.save(
        #         {
        #             "model": self._model_state_dict(),
        #             "optimizer": self.optimizer.state_dict(),
        #             "epoch": self.last_epoch
        #         },
        #         self.ckpt_dirpath / f"checkpoint_{self.last_epoch}.pth",
        #     )

    def _model_state_dict(self):
        """Returns state dict of model, taking care of DataParallel case."""
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()

    def save_best(self, ckpt_name="best"):
        """Saves only the best checkpoint. """

        torch.save(
            {
                "model": self._model_state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.last_epoch
            },
            self.ckpt_dirpath / f"checkpoint_{ckpt_name}.pth",
            _use_new_zipfile_serialization=False
        )

    def update_last_epoch(self, epoch=None):
        """ Update last epoch"""
        self.last_epoch = epoch
        self.logger.info(f'Setting the epoch number to {self.last_epoch}.')
        return


def load_checkpoint(checkpoint_pthpath):
    """Given a path to saved checkpoint, load corresponding state dicts
    of model and optimizer from it. This method checks if the current
    commit SHA of codebase matches the commit SHA recorded when this
    checkpoint was saved by checkpoint manager.

    Parameters
    ----------
    checkpoint_pthpath: str or pathlib.Path
        Path to saved checkpoint (as created by ``CheckpointManager``).

    Returns
    -------
    nn.Module, optim.Optimizer
        Model and optimizer state dicts loaded from checkpoint.
    """

    if isinstance(checkpoint_pthpath, str):
        checkpoint_pthpath = Path(checkpoint_pthpath)

    # load encoder, decoder, optimizer state_dicts
    components = torch.load(checkpoint_pthpath, map_location='cpu')
    # components = torch.load(checkpoint_pthpath,)
    return components["model"], components["optimizer"]


if __name__ == '__main__':
    pass
