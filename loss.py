from anatool import AnaLogger
import torch
import torch.nn as nn


def get_loss_criterion(opt, logger: AnaLogger, train_dataset):
    if opt.decoder == 'disc':
        criterion = nn.CrossEntropyLoss()
    elif opt.decoder == 'gen':
        criterion = nn.CrossEntropyLoss(
            ignore_index=train_dataset.vocabulary.PAD_INDEX
        )
    else:
        logger.error(f'Invalid parameter decoder {opt.decoder}.')
        raise NotImplementedError
    return criterion


def mse_loss(output, labels):
    batch_size, num_options = output.size()
    labels = labels.view(batch_size, -1)
    output = output.view(batch_size, -1)
    loss = torch.sum((output - labels) ** 2)
    return loss


def ndcg_loss(output, labels, log_softmax=nn.LogSoftmax(dim=-1)):
    output = log_softmax(output)
    batch_size, num_options = output.size()
    labels = labels.view(batch_size, -1)
    output = output.view(batch_size, -1)
    loss = -torch.mean(torch.sum(labels * output, dim=1))
    return loss


def criterion_loss(opt, batch, criterion, output):
    target = batch['ans_ind'] if opt.decoder != 'gen' else batch['ans_out']
    loss = criterion(
        output.view(-1, output.size(-1)),
        target.view(-1),
    )
    return loss
