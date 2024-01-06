import datetime
import random
import itertools
import os.path
from math import sqrt

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import relu
from torch.backends import cudnn
from tqdm import tqdm

from anatool import AnaArgParser, AnaLogger
from components import get_model, get_solver, get_dataloader
from visdialch.metrics import SparseGTMetrics, NDCG
from visdialch.utils.checkpointing import CheckpointManager, load_checkpoint
from loss import get_loss_criterion, mse_loss, ndcg_loss, criterion_loss


def train(opt, logger: AnaLogger, dataloader_dict, finetune=False, load_path='',
          finetune_regression=False, dense_scratch_train=False):
    """
    :param opt:
    :param logger:
    :param dataloader_dict:
    :param finetune:
    :param load_path:
    :param finetune_regression:
    :param dense_scratch_train: when we want to start training only on 2000 annotations
    :return:
    """
    train_dataset = dataloader_dict['train_dataset']
    train_dataloader = dataloader_dict['train_dataloader']
    val_dataset = dataloader_dict['val_dataset']
    val_dataloader = dataloader_dict['val_dataloader']
    eval_dataloader = dataloader_dict['eval_dataloader']

    model = get_model(opt=opt, logger=logger, train_dataset=train_dataset)
    if finetune and not dense_scratch_train:
        if load_path == '' or not os.path.exists(load_path):
            logger.error('Please provide a path for pre-trained model before starting fine tuning')
            raise FileNotFoundError
        logger.info('Begin finetuning:')

    optimizer, scheduler, iterations = get_solver(
        opt=opt, logger=logger, train_dataset=train_dataset,
        val_dataset=val_dataset, model=model, finetune=finetune
    )
    summary_writer = SummaryWriter(logdir=opt.exp_dir)
    checkpoint_manager = CheckpointManager(
        model=model, optimizer=optimizer, logger=logger,
        checkpoint_dirpath=os.path.join(opt.exp_dir, 'checkpoints')
    )
    sparse_metrics = SparseGTMetrics(logger=logger)
    ndcg = NDCG(logger=logger)
    best_val_loss = np.inf
    best_val_ndcg = 0.0

    # if loading from checkpoint, adjust start epoch and loaded parameters.

    # 1. if finetuning -> load from saved model,
    # 2. train -> default load_path = '',
    # 3. else load pthpath.
    if (not finetune and load_path == '') or dense_scratch_train:
        start_epoch = 1
    else:
        # path example /checkpoint_xx.pth.
        # to cater model finetuning from models with 'best_ndcg' checkpoint.
        try:
            start_epoch = int(load_path.split('_')[-1][:-4]) + 1
        except:
            start_epoch = 1

        model_state_dict, optimizer_state_dict = load_checkpoint(checkpoint_pthpath=load_path)

        checkpoint_manager.update_last_epoch(start_epoch)

        if isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)

        # for finetuning optimizer should start from its learning rate.
        if not finetune:
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            logger.info('Optimizer not loaded. Different optimizer for finetuning')
        logger.info(f'Loaded model from {load_path}.')

    # training loop
    # forever increasing counter to keep track of iterations (for tensorboard log).
    global_iteration_step = (start_epoch - 1) * iterations

    running_loss = 0.0
    train_begin = datetime.datetime.now()

    criterion = None
    if finetune:
        end_epoch = start_epoch + opt.num_epochs_curriculum
        if finetune_regression:
            criterion = nn.MultiLabelSoftMarginLoss()
    else:
        end_epoch = opt.num_epochs
        criterion = get_loss_criterion(
            opt=opt,
            logger=logger,
            train_dataset=train_dataset
        )
    for epoch in range(start_epoch, end_epoch + 1):
        train_dataloader.sampler.set_epoch(epoch=epoch)
        val_dataloader.sampler.set_epoch(epoch=epoch)
        # combine dataloaders if training on train + val.
        if opt.training_splits == 'trainval':
            combined_dataloader = itertools.chain(train_dataloader, val_dataloader)
        elif opt.training_splits == 'train':
            combined_dataloader = itertools.chain(train_dataloader)
        # for testing code, since val set is smaller
        else:
            combined_dataloader = itertools.chain(val_dataloader)

        logger.info(f'Training for epoch {epoch}')
        for i, batch in enumerate(tqdm(combined_dataloader)):
            for key in batch:
                batch[key] = batch[key].cuda()
            optimizer.zero_grad()
            output = model(batch, False, False)

            if finetune:
                target = batch['gt_relevance']
                # same as for ndcg validation, only one round is present.
                output = output[
                         torch.arange(output.size(0)),
                         batch['round_id'] - 1,
                         :]
                if finetune_regression:
                    batch_loss = mse_loss(output=output, labels=target)
                else:
                    batch_loss = ndcg_loss(output=output, labels=target)
            else:
                batch_loss = criterion_loss(opt=opt, batch=batch, criterion=criterion, output=output)

                is_cfq, is_cfi = False, False
                margin_value = 0
                if opt.cfq_interval != -1 and i % opt.cfq_interval == 0:
                    is_cfq = True
                    margin_value = (opt.cfq_interval / 10) ** 2
                elif opt.cfi_interval != -1 and i % opt.cfi_interval == 0:
                    is_cfi = True
                    margin_value = (opt.cfi_interval / 10) ** 2
                cs_loss = 0
                if is_cfq:
                    cfq_output = model(batch, True, False)
                    cfq_loss = criterion_loss(opt=opt, batch=batch, criterion=criterion,
                                              output=cfq_output)
                    cs_loss = opt.lambda_cfq * cfq_loss
                if is_cfi:
                    cfic_output = model(batch, False, True)
                    cfic_loss = criterion_loss(opt=opt, batch=batch, criterion=criterion,
                                               output=cfic_output)
                    cs_loss = opt.lambda_cfi * cfic_loss
                if (is_cfq or is_cfi) and (batch_loss + cs_loss + margin_value).item() >= 0:
                    batch_loss = batch_loss + cs_loss + margin_value

            batch_loss.backward()

            optimizer.step()

            # update running loss and decay learning rates.
            if running_loss > 0.0:
                running_loss = 0.95 * running_loss + 0.05 * batch_loss.item()
            else:
                running_loss = batch_loss.item()

            # lambda_lr was configured to reduce lr after milestone epoch.
            # if lr_scheduler_type == 'lambda_lr':
            scheduler.step()

            global_iteration_step += 1

            if global_iteration_step % 100 == 0:
                logger.info("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:8f}]".format(
                    datetime.datetime.now() - train_begin, epoch,
                    global_iteration_step, running_loss,
                    optimizer.param_groups[0]['lr']))
                summary_writer.add_scalar(
                    tag='train/loss',
                    scalar_value=batch_loss,
                    global_step=global_iteration_step
                )
                summary_writer.add_scalar(
                    tag='train/lr',
                    scalar_value=optimizer.param_groups[0]['lr'],
                    global_step=global_iteration_step
                )
        torch.cuda.empty_cache()

        # on epoch end (checkpointing and validation).
        if not finetune:
            checkpoint_manager.step(epoch=epoch)
        else:
            logger.info('Validating before checkpointing')

        if opt.validate and opt.local_rank == 0:
            # switch dropout, batchnorm and so on to the correct mode.
            model.eval()
            val_loss = 0

            logger.info(f'\nValidation after epoch {epoch}:')
            for i, batch in enumerate(tqdm(eval_dataloader)):
                for key in batch:
                    batch[key] = batch[key].cuda()
                with torch.no_grad():
                    output = model(batch)
                    if finetune:
                        target = batch['gt_relevance']
                        out_ndcg = output[
                                   torch.arange(output.size(0)),
                                   batch['round_id'] - 1,
                                   :]
                        if finetune_regression:
                            val_batch_loss = mse_loss(output=out_ndcg, labels=target)
                        else:
                            val_batch_loss = ndcg_loss(output=out_ndcg, labels=target)
                    else:
                        val_batch_loss = criterion_loss(opt=opt, batch=batch, criterion=criterion, output=output)

                summary_writer.add_scalar(
                    tag='val/loss',
                    scalar_value=val_batch_loss,
                    global_step=global_iteration_step
                )
                val_loss += val_batch_loss.item()

                sparse_metrics.observe(predicted_scores=output, target_ranks=batch['ans_ind'])

                if 'gt_relevance' in batch:
                    output = output[
                             torch.arange(output.size(0)),
                             batch['round_id'] - 1,
                             :]
                    ndcg.observe(output, batch['gt_relevance'])

            all_metrics = {}
            all_metrics.update(sparse_metrics.retrieve(reset=True, get_last_num_round=True))
            all_metrics.update(ndcg.retrive(reset=True))
            logger.info(f'{opt.local_rank} metrics')
            for metirc_name, metric_value in all_metrics.items():
                logger.info(f'{metirc_name}: {metric_value}')
            summary_writer.add_scalars(
                main_tag='metrics',
                tag_scalar_dict=all_metrics,
                global_step=global_iteration_step
            )

            model.train()
            torch.cuda.empty_cache()

            val_loss = val_loss / len(val_dataloader)
            logger.info(f'Validation loss for epoch {epoch} is {val_loss}.\n')

            if val_loss < best_val_loss:
                logger.info(f'Best model found at epoch {epoch}! Saving now.')
                best_val_loss = val_loss
                checkpoint_manager.save_best()
            else:
                logger.info(f'Not saving the model at epoch {epoch}!')

            val_ndcg = all_metrics['ndcg']
            if val_ndcg > best_val_ndcg:
                logger.info(f'Best ndcg model found at epoch {epoch}! Saving now.')
                best_val_ndcg = val_ndcg
                checkpoint_manager.save_best(ckpt_name='best_ndcg')
            else:
                logger.info(f'Not saving the model at epoch {epoch}!')
        break


def main():
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    logger.info(f'Starting the model run now: {datetime.datetime.now()}')
    logger.info(opt)

    torch.manual_seed(opt.local_rank)
    torch.cuda.manual_seed_all(opt.local_rank)
    np.random.seed(opt.local_rank)
    random.seed(opt.local_rank)
    cudnn.benchmark = False
    cudnn.deterministic = True

    torch.cuda.set_device(opt.local_rank)
    dist.init_process_group(backend='nccl')

    logger.info(
        f'Running on: {opt.local_rank}\n' +
        f'Training phase from the python code: {opt.phase}'
    )

    # normal training
    if opt.phase in ['training', 'both']:
        logger.info('Starting training.')
        dataloader_dict = get_dataloader(
            opt=opt,
            logger=logger,
            finetune=False,
        )
        train(
            opt=opt, logger=logger, dataloader_dict=dataloader_dict,
            finetune=False, load_path=opt.load_path
        )

    # Sequential since you can ask for both train and finetune.
    if opt.phase in ['finetuning', 'both']:
        logger.info('Starting finetuning.')
        finetune_path = opt.load_finetune_path \
            if opt.phase == 'finetuning' \
            else os.path.join(opt.exp_dir, 'checkpoints', 'checkpoint_best_ndcg.pth')
        dataloader_dict = get_dataloader(
            opt=opt,
            logger=logger,
            finetune=True
        )
        train(
            opt=opt, logger=logger, dataloader_dict=dataloader_dict,
            finetune=True, load_path=finetune_path,
            finetune_regression=opt.dense_regression
        )

    # Train only on dense annotations.
    if opt.phase in ['dense_scratch_train']:
        logger.info('Starting finetuning.')
        dataloader_dict = get_dataloader(
            opt=opt,
            logger=logger,
            finetune=True
        )
        train(
            opt=opt, logger=logger, dataloader_dict=dataloader_dict,
            finetune=True, dense_scratch_train=True
        )

    logger.info(f'Training done! Time: {datetime.datetime.now()}')


if __name__ == '__main__':
    torch.cuda.reset_max_memory_allocated()
    main()
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak Memory: {peak_memory / (1024 ** 2):.2f} MB")
