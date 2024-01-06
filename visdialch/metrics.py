"""
A Metric observes output of certain model, for example, in form of logits or
scores, and accumulates a particular metric with reference to some provided
targets. In context of VisDial, we use Recall (@ 1, 5, 10), Mean Rank, Mean
Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG).

Each ``Metric`` must atleast implement three methods:
    - ``observe``, update accumulated metric with currently observed outputs
      and targets.
    - ``retrieve`` to return the accumulated metric., an optionally reset
      internally accumulated metric (this is commonly done between two epochs
      after validation).
    - ``reset`` to explicitly reset the internally accumulated metric.

Caveat, if you wish to implement your own class of Metric, make sure you call
``detach`` on output tensors (like logits), else it will cause memory leaks.
"""
import torch

from anatool import AnaArgParser, AnaLogger


def scores_to_ranks(scores: torch.Tensor):
    """
    convert model output scores into ranks
    """
    batch_size, num_rounds, num_options = scores.size()
    scores = scores.view(-1, num_options)

    # sort in descending order - largest score gets highest rank.
    sorted_ranks, ranked_idx = scores.sort(dim=1, descending=True)

    # i-th position in ranked_idx specifies which score shall take this
    # position, but we want i-th position to have rank of score at that
    # position, do this conversion.
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(num_options):
            ranks[i][ranked_idx[i][j]] = j
    # convert from 0-99 ranks to 1-100 ranks.
    ranks += 1
    ranks = ranks.view(batch_size, num_rounds, num_options)
    return ranks


class SparseGTMetrics:
    """
    A class to accumulate all metrics with sparse ground truth annotations.
    These include Recall (@ 1, 5, 10), Mean Rank and Mean Reciprocal Rank.
    """

    def __init__(self, logger: AnaLogger):
        self._rank_list = []
        self.logger = logger

    def observe(self, predicted_scores: torch.Tensor, target_ranks: torch.Tensor):
        predicted_scores = predicted_scores.detach()
        # shape: (batch_size, num_rounds, num_options)
        predicted_ranks = scores_to_ranks(predicted_scores)
        batch_size, num_rounds, num_options = predicted_ranks.size()

        predicted_ranks = predicted_ranks.view(
            batch_size * num_rounds, num_options
        )

        # shape: (batch_size * num_rounds, )
        target_ranks = target_ranks.view(batch_size * num_rounds).long()
        # shape: (batch_size * num_rounds, )
        predicted_gt_ranks = predicted_ranks[
            torch.arange(batch_size * num_rounds), target_ranks
        ]

        self._rank_list.extend(list(predicted_gt_ranks.cpu().numpy()))

    def retrieve(self, reset=True, get_last_num_round=False):
        num_examples = len(self._rank_list)
        if num_examples > 0:
            if get_last_num_round:
                __rank_list = torch.tensor(self._rank_list[9:len(self._rank_list):10]).float()
            else:
                __rank_list = torch.tensor(self._rank_list).float()
            metrics = {
                'r@1': torch.mean((__rank_list <= 1).float()).item() * 100,
                'r@5': torch.mean((__rank_list <= 5).float()).item() * 100,
                'r@10': torch.mean((__rank_list <= 10).float()).item() * 100,
                'mean': torch.mean(__rank_list).item(),
                'mrr': torch.mean(__rank_list.reciprocal()).item() * 100,
            }
        else:
            metrics = {}

        if reset:
            self._rank_list = []
        return metrics


class NDCG:
    def __init__(self, logger: AnaLogger, is_direct_ranks=False):
        """
        param is_direct_ranks: if we pass directly ranks instead of scores in observe
        """
        self.logger = logger
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0
        self.is_direct_ranks = is_direct_ranks

    def observe(self, predicted_scores: torch.Tensor, target_relevance: torch.Tensor):
        """
        Observe model output scores and target ground truth relevance and
        accumulate NDCG metric.

        Parameters
        ----------
        predicted_scores: torch.Tensor
            A tensor of shape (batch_size, num_options), because dense
            annotations are available for 1 randomly picked round out of 10.
        target_relevance: torch.Tensor
            A tensor of shape same as predicted scores, indicating ground truth
            relevance of each answer option for a particular round.
        """
        if not self.is_direct_ranks:
            predicted_scores = predicted_scores.detach()

            # shape: (batch_size, 1, num_options)
            predicted_scores = predicted_scores.unsqueeze(1)
            predicted_ranks = scores_to_ranks(predicted_scores)

            # shape: (batch_size, num_options)
            predicted_ranks = predicted_ranks.squeeze(1)
        else:
            # now ranks are passed instead of scores.
            assert len(predicted_scores.size()) == 2  # (batch_size, num_options)
            predicted_ranks = predicted_scores

        batch_size, num_options = predicted_ranks.size()

        k = torch.sum((target_relevance != 0).long(), dim=-1)

        # shape: (batch_size, num_options)
        _, rankings = torch.sort(input=predicted_ranks, dim=-1)
        # sort relevance in descending order so highest relevance gets top rank
        _, best_rankings = torch.sort(input=target_relevance, dim=-1, descending=True)

        # shape: (batch_size, )
        batch_ndcg = []
        for batch_idx in range(batch_size):
            num_relevant = k[batch_idx]
            dcg = self.dcg(
                rankings=rankings[batch_idx][:num_relevant],
                relevance=target_relevance[batch_idx]
            )
            best_dcg = self.dcg(
                rankings=best_rankings[batch_idx][:num_relevant],
                relevance=target_relevance[batch_idx]
            )
            batch_ndcg.append(dcg / best_dcg)

        self._ndcg_denominator += batch_size
        self._ndcg_numerator += sum(batch_ndcg)

    @staticmethod
    def dcg(rankings: torch.Tensor, relevance: torch.Tensor):
        sorted_relevance = relevance[rankings].cpu().float()
        discounts = torch.log2(input=torch.arange(len(rankings)).float() + 2)
        # log2(i+1) to cater 0-indexing add extra 1
        return torch.sum(sorted_relevance / discounts, dim=-1)

    def retrive(self, reset=True):
        if self._ndcg_denominator > 0:
            metrics = {
                'ndcg': float(self._ndcg_numerator / self._ndcg_denominator) * 100
            }
        else:
            metrics = {}

        if reset:
            self._ndcg_denominator = 0.0
            self._ndcg_numerator = 0.0
        return metrics


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    ranks = scores_to_ranks(
        scores=torch.rand([3, 2, 3], dtype=torch.float)
    )
    target_ranks = torch.ones(3, 2)
    sm = SparseGTMetrics(logger=logger)
    sm.observe(predicted_scores=ranks, target_ranks=target_ranks)
    sm.retrieve()
    a = torch.rand(4, 5)
    b = torch.arange(4)
    c = torch.tensor([0, 1, 2, 3])
    print(a)
    print(a[b, :])
    print(c)
    print(a[b, c])
