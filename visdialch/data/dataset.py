import json
from typing import List

import torch
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from anatool import AnaLogger, AnaArgParser
from visdialch.data.readers import DialogsReader, DenseAnnotationsReader, ImageFeaturesReader
from visdialch.data.vocabulary import Vocabulary


class VisDialDataset(Dataset):
    """
    A full representation of VisDial v1.0 (train/val/test) dataset. According
    to the appropriate split, it returns dictionary of question, image,
    history, caption, ground truth answer, answer options, dense annotations etc.
    """

    def __init__(
            self,
            opt,
            logger: AnaLogger,
            dialogs_json_path,
            dense_annotations_json_path=None,
            finetune=False,
            overfit=False,
            in_memory=False,
            return_options=True,
            add_boundary_toks=False,

    ):
        super(VisDialDataset, self).__init__()
        self.logger = logger
        self.opt = opt
        self.return_options = return_options
        self.finetune = finetune
        self.add_boundary_toks = add_boundary_toks
        self.dialogs_reader = DialogsReader(
            logger=logger,
            dialogs_json_path=dialogs_json_path,
            num_examples=(10 if overfit else None),
        )

        # if finetuning for train/val otherwise just val set
        if self.finetune or ('val' in self.split and dense_annotations_json_path is not None):
            self.annotations_reader = DenseAnnotationsReader(
                dense_annotations_json_path=dense_annotations_json_path,
                logger=self.logger,
            )
        else:
            self.annotations_reader = None

        self.vocabulary = Vocabulary(
            logger=self.logger,
            word_counts_path=opt.word_counts_json,
            min_count=opt.vocab_min_count
        )

        image_features_hdf_path = opt.image_features_train_h5
        if 'val' in self.dialogs_reader.spilt:
            image_features_hdf_path = opt.image_features_val_h5
        elif 'test' in self.dialogs_reader.spilt:
            image_features_hdf_path = opt.image_features_test_h5

        self.hdf_reader = ImageFeaturesReader(
            logger=self.logger,
            features_hdf_path=image_features_hdf_path,
            in_memory=in_memory
        )

        # keep a list of image_ids as primary keys to access data
        # for finetune we use only those image id where we have dense annotations
        if self.finetune:
            self.image_ids = list(self.annotations_reader.keys)
        else:
            self.image_ids = list(self.dialogs_reader.dialogs.keys())

        if overfit:
            self.image_ids = self.image_ids[:10]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # get image_id, which serves as a primary key for current instance
        image_id = self.image_ids[index]

        # get image features for this image_id using hdf reader
        image_features = self.hdf_reader[image_id]
        image_features = torch.tensor(image_features)

        # normalize image features at zero-th dimension (since there's no batch dimension)
        if self.opt.img_norm:
            image_features = normalize(image_features, dim=0, p=2)

        # retrieve instance for this image_id using json reader
        visdial_instance = self.dialogs_reader[image_id]
        caption = visdial_instance['caption']

        dialog = visdial_instance['dialog']

        for i in range(len(dialog)):

            dialog[i]['question'] = self.vocabulary.to_indices(
                dialog[i]['question']
            )

            if self.add_boundary_toks:
                dialog[i]['answer'] = self.vocabulary.to_indices(
                    [self.vocabulary.SOS_TOKEN] +
                    dialog[i]['answer'] +
                    [self.vocabulary.EOS_TOKEN]
                )
            else:
                dialog[i]['answer'] = self.vocabulary.to_indices(
                    dialog[i]['answer']
                )

            # for disc decoder
            if self.return_options:
                # ideally should be in if-else clause

                for j in range(len(dialog[i]['answer_options'])):

                    if self.add_boundary_toks:
                        dialog[i]['answer_options'][j] = self.vocabulary.to_indices(
                            [self.vocabulary.SOS_TOKEN] +
                            dialog[i]['answer_options'][j] +
                            [self.vocabulary.EOS_TOKEN]
                        )
                    else:
                        dialog[i]['answer_options'][j] = self.vocabulary.to_indices(
                            dialog[i]['answer_options'][j]
                        )

        questions, question_lengths = self._pad_sequences(
            sequences=[dialog_round['question'] for dialog_round in dialog]
        )

        # convert word tokens of caption, question, answer and answer option to integers
        caption = self.vocabulary.to_indices(caption)
        # caption = caption[:self.opt.max_cap_sequence_length - 1] + [self.vocabulary.EOS_INDEX]
        caption = caption[:self.opt.max_cap_sequence_length]
        caption_content = self._pad_sequences(
            sequences=[caption],
            is_cap=True
        )
        caption, caption_length = caption_content[0][0], caption_content[-1][0]
        history, history_lengths = self._get_history(
            questions=[dialog_round["question"] for dialog_round in dialog],
            answers=[dialog_round["answer"] for dialog_round in dialog]
        )
        # exclude the last answer
        answer_in, answer_lengths = self._pad_sequences(
            sequences=[dialog_round['answer'][:-1] for dialog_round in dialog]
        )
        answer_out, answer_lengths = self._pad_sequences(
            sequences=[dialog_round['answer'][1:] for dialog_round in dialog]
        )

        item = {
            'img_ids': torch.tensor(image_id).long(),
            'img_feat': image_features,
            'cap': caption.long(),
            'cap_len': torch.tensor(caption_length).long(),
            'ques': questions.long(),
            'ques_len': torch.tensor(question_lengths).long(),
            'hist': history.long(),
            'hist_len': torch.tensor(history_lengths).long(),
            'ans_in': answer_in.long(),
            'ans_out': answer_out.long(),
            'ans_len': torch.tensor(answer_lengths).long(),
            'num_rounds': torch.tensor(visdial_instance['num_rounds']).long()
        }

        if self.return_options:
            if self.add_boundary_toks:
                answer_options_in, answer_options_out = [], []
                answer_option_lengths = []
                for dialog_round in dialog:
                    options, option_lengths = self._pad_sequences(
                        sequences=[
                            option[:-1]
                            for option in dialog_round['answer_options']
                        ]
                    )
                    answer_options_in.append(options)
                    options, _ = self._pad_sequences(
                        [
                            option[1:]
                            for option in dialog_round['answer_options']
                        ]
                    )
                    answer_options_out.append(options)
                    answer_option_lengths.append(option_lengths)
                answer_options_in = torch.stack(answer_options_in, 0)
                answer_options_out = torch.stack(answer_options_out, 0)
                item.update({
                    'opt_in': answer_options_in.long(),
                    'opt_out': answer_options_out.long(),
                    'opt_len': torch.tensor(answer_option_lengths).long()
                })
            else:
                answer_options = []
                answer_option_lengths = []
                for dialog_round in dialog:
                    options, option_lengths = self._pad_sequences(
                        sequences=dialog_round['answer_options']
                    )
                    answer_options.append(options)
                    answer_option_lengths.append(option_lengths)
                answer_options = torch.stack(answer_options, 0)
                # used by disc model
                # option_lengths: used by model to select non-zero options
                item.update({
                    'opt': answer_options.long(),
                    'opt_len': torch.tensor(answer_option_lengths).long()
                })

            if 'test' not in self.split:
                answer_indices = [
                    dialog_round['gt_index'] for dialog_round in dialog
                ]
                # used by evaluation for NDCG
                item['ans_ind'] = torch.tensor(answer_indices).long()

        # gather dense annotations
        if self.finetune or ('val' in self.split):
            dense_annotations = self.annotations_reader[image_id]

            # have to do this because of changed dict key in train
            if 'val' in self.split:
                item['gt_relevance'] = torch.tensor(
                    dense_annotations['gt_relevance']
                ).float()
            elif 'train' in self.split:
                item['gt_relevance'] = torch.tensor(
                    dense_annotations['relevance']
                ).float()

            item['round_id'] = torch.tensor(
                dense_annotations['round_id']
            ).long()

        return item

    @property
    def split(self):
        return self.dialogs_reader.spilt

    def _pad_sequences(self, sequences: List[List[int]], is_cap=False):
        """Given tokenized sequences (either questions, answers or answer
        options, tokenized in ``__getitem__``), padding them to maximum
        specified sequence length. Return as a tensor of size
        ``(*, max_sequence_length)``.

        This method is only called in ``__getitem__``, chunked out separately
        for readability.

        Parameters
        ----------
        sequences : list[list[int]]
            List of tokenized sequences, each sequence is typically a
            List[int].

        Returns
        -------
        torch.Tensor, torch.Tensor
            Tensor of sequences padded to max length, and length of sequences
            before padding.
        """
        max_sequence_length = self.opt.max_cap_sequence_length if is_cap else self.opt.max_sequence_length
        for i in range(len(sequences)):
            sequences[i] = sequences[i][:max_sequence_length - 1]

        sequence_lengths = [len(sequence) for sequence in sequences]

        # pad all sequences to max_sequence_length
        max_padded_sequences = torch.full(
            (len(sequences), max_sequence_length),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_sequences = pad_sequence(
            [torch.tensor(sequence) for sequence in sequences],
            batch_first=True,
            padding_value=self.vocabulary.PAD_INDEX,
        )
        max_padded_sequences[:, :padded_sequences.size(1)] = padded_sequences
        return max_padded_sequences, sequence_lengths

    def _get_history(self, questions: List[List[int]], answers: List[List[int]]):
        for i in range(len(questions)):
            questions[i] = questions[i][
                           :self.opt.max_sequence_length - 1
                           ]
        for i in range(len(answers)):
            answers[i] = answers[i][
                         :self.opt.max_sequence_length - 1
                         ]
        # TODO EOS or SOS or Nothing
        # history = [[self.vocabulary.EOS_INDEX]]
        history = []
        for question, answer in zip(questions, answers):
            history.append(question + answer + [self.vocabulary.EOS_INDEX])

        # drop last entry from history (there's no 11th question)
        # the 10th answer is not included in history
        history = history[:-1]
        history.append(questions[-1] + [self.vocabulary.EOS_INDEX])
        max_history_length = self.opt.max_sequence_length * 2

        history_lengths = [len(round_history) for round_history in history]

        max_padded_history = torch.full(
            (len(history), max_history_length),
            fill_value=self.vocabulary.PAD_INDEX
        )

        padded_history = pad_sequence(
            [torch.tensor(round_history) for round_history in history],
            batch_first=True,
            padding_value=self.vocabulary.PAD_INDEX
        )
        max_padded_history[:, :padded_history.size(1)] = padded_history
        return max_padded_history, history_lengths


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    vdd = VisDialDataset(
        opt=opt,
        logger=logger,
        dialogs_json_path=opt.val_json,
        dense_annotations_json_path=opt.val_dense_json
    )
    test_json_path = opt.test_json
    f = open(test_json_path, 'r')
    content = json.loads(f.read())
    'the group of people are playing soccer on the field'
    answers = content['data']['answers']
    questions = content['data']['questions']
    print(questions[41755])
    print(answers[4801])

    print(questions[10131])
    print(answers[23770])

    print(questions[45230])
    print(answers[2418])

    print(questions[3122])
    print(answers[5123])

    print(questions[38137])
    print(answers[24288])

    print(questions[19360])
    # logger.debug(vdd[0])
