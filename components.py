import os
from bisect import bisect

import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from anatool import AnaLogger
from visdialch.data.dataset import VisDialDataset
from visdialch.decoders import decoders
from visdialch.encoders import encoders
from visdialch.model import EncoderDecoderModel


def get_dataloader(opt, logger: AnaLogger, finetune=False):
    logger.info(f'Pin memory is set to {opt.pin_memory}.')
    train_set = VisDialDataset(
        opt=opt,
        logger=logger,
        dialogs_json_path=opt.train_json,
        dense_annotations_json_path=opt.train_dense_json,
        finetune=finetune,
        overfit=opt.overfit,
        in_memory=opt.pin_memory,
        return_options=True if opt.decoder != 'gen' else False,
        add_boundary_toks=False if opt.decoder != 'gen' else True,
    )
    train_sampler = DistributedSampler(train_set)
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        num_workers=opt.cpu_workers,
        pin_memory=opt.pin_memory,
        sampler=train_sampler
    )
    val_set = VisDialDataset(
        opt=opt,
        logger=logger,
        dialogs_json_path=opt.val_json,
        dense_annotations_json_path=opt.val_dense_json,
        finetune=finetune,
        overfit=opt.overfit,
        in_memory=opt.pin_memory,
        return_options=True,
        add_boundary_toks=False if opt.decoder != 'gen' else True
    )
    val_sampler = DistributedSampler(val_set)
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=opt.batch_size if opt.decoder != 'gen' else 5,
        num_workers=opt.cpu_workers,
        pin_memory=opt.pin_memory,
        sampler=val_sampler
    )
    eval_dataloader = DataLoader(
        dataset=val_set,
        batch_size=opt.eval_batch_size if opt.decoder != 'gen' else 5,
        num_workers=opt.cpu_workers,
        pin_memory=opt.pin_memory,
        shuffle=False
    )

    dataloader_dict = {
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'train_dataset': train_set,
        'val_dataset': val_set,
        'eval_dataloader': eval_dataloader
    }
    return dataloader_dict


def get_model(opt, logger: AnaLogger, train_dataset):
    encoder = encoders(
        opt=opt,
        logger=logger,
        vocabulary=train_dataset.vocabulary
    )
    decoder = decoders(
        opt=opt,
        logger=logger,
        vocabulary=train_dataset.vocabulary
    )
    logger.info(f'Encoder: {opt.encoder}.')
    logger.info(f'Decoder: {opt.decoder}.')

    # initializing word embed using GloVe.
    if opt.glove_npy != '' and os.path.exists(opt.glove_npy):
        encoder.word_embed.weight.data = torch.from_numpy(np.load(opt.glove_npy))
        logger.info(f'Loaded glove vectors from: {opt.glove_npy}.')

    # share word embedding between encoder and decoder.
    if encoder.word_embed and decoder.word_embed:
        decoder.word_embed = encoder.word_embed

    model = EncoderDecoderModel(encoder, decoder).cuda()
    model = DistributedDataParallel(module=model, device_ids=[opt.local_rank])
    return model


def get_solver(opt, logger: AnaLogger, train_dataset, val_dataset, model, finetune=False):
    initial_lr = opt.initial_lr_curriculum if finetune else opt.initial_lr
    if torch.cuda.device_count() > 0:
        nodes = torch.cuda.device_count()
    else:
        nodes = 1
    if opt.training_splits == 'trainval':
        iterations = 1 + (len(train_dataset) + len(val_dataset)) // (opt.batch_size * nodes)
    else:
        iterations = 1 + len(train_dataset) // (opt.batch_size * nodes)

    def lr_lambda_fun(current_iteration) -> float:
        """Returns a learning rate multiplier.

        Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
        and then gets multiplied by `lr_gamma` every time a milestone is crossed.
        """
        current_epoch = float(current_iteration) / iterations
        if current_epoch <= opt.warmup_epochs:
            alpha = current_epoch / float(opt.warmup_epochs)
            return opt.warmup_factor * (1.0 - alpha) + alpha
        else:
            idx = bisect(opt.lr_milestones, current_epoch)
            return pow(opt.lr_gamma, idx)

    logger.info(f'Initial LR set to: {initial_lr}.')
    optimizer = optim.Adamax(model.parameters(), lr=initial_lr)

    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda_fun)

    return optimizer, scheduler, iterations
