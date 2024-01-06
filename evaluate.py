import datetime
import random
import json
import os.path
import pickle as pkl

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from anatool import AnaArgParser, AnaLogger
from visdialch.data.dataset import VisDialDataset
from visdialch.decoders import decoders
from visdialch.encoders import encoders
from visdialch.metrics import SparseGTMetrics, NDCG, scores_to_ranks
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import load_checkpoint


def evaluate(opt, logger: AnaLogger, eval_dataset, eval_dataloader):
    encoder = encoders(
        opt=opt,
        logger=logger,
        vocabulary=eval_dataset.vocabulary
    )
    decoder = decoders(
        opt=opt,
        logger=logger,
        vocabulary=eval_dataset.vocabulary
    )
    logger.info(f'Encoder: {opt.encoder}.')
    logger.info(f'Decoder: {opt.decoder}.')

    decoder.word_embed = encoder.word_embed

    model = EncoderDecoderModel(encoder, decoder).cuda()
    model = DataParallel(module=model, device_ids=opt.gpu_ids)
    model_state_dict, _ = load_checkpoint(opt.load_path)

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    logger.info(f'Loaded model from {opt.load_path}.')

    # declare metric accumulators (won't be used if split==test).
    sparse_metrics = SparseGTMetrics(logger=logger)
    ndcg = NDCG(logger=logger)

    model.eval()
    ranks_json = []
    # saving log probs for ensembling
    opt_log_probs = []
    batch_element_list = []

    for batch_num, batch in enumerate(tqdm(eval_dataloader)):
        for key in batch:
            batch[key] = batch[key].cuda()
        with torch.no_grad():
            output = model(batch)

        # output shape: (bs, rounds, options)
        ranks = scores_to_ranks(output)

        log_softmax = nn.LogSoftmax(dim=-1)
        softmax_probs = log_softmax(output)
        log_probs = output.view(-1, 10, 100).cpu().numpy()
        softmax_probs = softmax_probs.view(-1, 10, 100).cpu().numpy()

        for i in range(len(batch['img_ids'])):
            # Cast into types explicitly to ensure no errors in schema.
            # Round ids are 1-10, not 0-9.

            if opt.split == 'test':
                ranks_json.append(
                    {
                        'image_id': batch['img_ids'][i].item(),
                        'round_id': int(batch['num_rounds'][i].item()),
                        'ranks': [
                            rank.item()
                            for rank in ranks[i][batch['num_rounds'][i] - 1]
                        ]
                    }
                )
                opt_log_probs.append(list(log_probs[i][batch['num_rounds'][i] - 1]))
            else:
                for j in range(batch["num_rounds"][i]):
                    ranks_json.append(
                        {
                            "image_id": batch["img_ids"][i].item(),
                            "round_id": int(j + 1),
                            "ranks": [rank.item() for rank in ranks[i][j]],
                        }
                    )
                # num_rounds will be 10 for val, however round_id is used for dense
                # careful: round_id -> 1-index
                if 'gt_relevance' in batch:
                    opt_log_probs.append(
                        {
                            'image_id': batch['img_ids'][i].item(),
                            'round_id': int(batch['round_id'][i].item()),
                            'log_probs': list(log_probs[i][batch['round_id'][i] - 1]),
                            'softmax_probs': list(softmax_probs[i][batch['round_id'][i] - 1])
                        }
                    )
        if opt.split == 'val':
            batch_element = {
                'ans_ind': batch['ans_ind'].cpu().numpy(),
                'round_id': batch['round_id'].cpu().numpy(),
                'output': output.cpu().numpy()
            }

            sparse_metrics.observe(predicted_scores=output, target_ranks=batch['ans_ind'])
            if 'gt_relevance' in batch:
                output = output[
                         torch.arange(output.size(0)),
                         batch['round_id'] - 1,
                         :]
                ndcg.observe(predicted_scores=output, target_relevance=batch['gt_relevance'])

                batch_element['output_gt_relevance'] = output.cpu().numpy()
                batch_element['gt_relevance'] = batch['gt_relevance'].cpu().numpy()
                batch_element['img_ids'] = batch['img_ids'].cpu().numpy()

            batch_element_list.append(batch_element)

        break

    logger.info(f'Total batches considered: {len(eval_dataloader)}')

    if opt.split == 'val':
        all_metrics = {}
        all_metrics.update(sparse_metrics.retrieve(reset=True, get_last_num_round=True))
        all_metrics.update(ndcg.retrive(reset=True))
        for metric_name, metric_value in all_metrics.items():
            logger.info(f'{metric_name}: {metric_value}')

    save_ranks_path = os.path.join(opt.exp_dir, 'val_ranks.json')
    logger.info(f'Writing ranks to {save_ranks_path}.')
    json.dump(ranks_json, open(save_ranks_path, 'w'))

    # For test we save np array for ensembling,
    # for val the whole json as annotations --> for visualization
    save_opt_log_probs_path = os.path.join(opt.exp_dir, 'opt_log_probs.pkl')

    if opt.split == 'test':
        opt_log_probs = np.array(opt_log_probs)
        logger.debug(opt_log_probs.shape)

    with open(save_opt_log_probs_path, 'wb') as fp:
        pkl.dump(opt_log_probs, fp)
    logger.info(f'Saving the output log probs to {save_opt_log_probs_path}.')

    # To get oracle preds
    if opt.split == 'val':
        save_batch_element_path = os.path.join(opt.exp_dir, 'batch_element.pkl')
        with open(save_batch_element_path, 'wb') as fp:
            pkl.dump(batch_element_list, fp)
        logger.info(f'Saving the output and batch to {save_batch_element_path}.')


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

    logger.info(f'Running on: {opt.gpu_ids}')
    torch.cuda.set_device(device=torch.device('cuda', opt.gpu_ids[0]))

    logger.info('Starting evaluation.')
    if opt.split == 'val':
        eval_dataset = VisDialDataset(
            opt=opt,
            logger=logger,
            dialogs_json_path=opt.val_json,
            dense_annotations_json_path=opt.val_dense_json,
            overfit=opt.overfit,
            in_memory=opt.pin_memory,
            return_options=True,
            add_boundary_toks=False if opt.decoder != 'gen' else True
        )
    else:
        eval_dataset = VisDialDataset(
            opt=opt,
            logger=logger,
            dialogs_json_path=opt.test_json,
            overfit=opt.overfit,
            in_memory=opt.pin_memory,
            return_options=True,
            add_boundary_toks=False if opt.decoder != 'gen' else True
        )

    eval_dataloder = DataLoader(
        dataset=eval_dataset,
        batch_size=opt.batch_size if opt.decoder != 'gen' else 5,
        num_workers=opt.cpu_workers,
        pin_memory=opt.pin_memory,
    )
    evaluate(
        opt=opt,
        logger=logger,
        eval_dataset=eval_dataset,
        eval_dataloader=eval_dataloder
    )
    logger.info(f'Evaluation done! Time: {datetime.datetime.now()}')


if __name__ == '__main__':
    main()
