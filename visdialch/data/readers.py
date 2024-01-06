import copy
import json
import os
from typing import Dict, Any, Set, Union

import h5py
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from anatool import AnaLogger, AnaArgParser


class DialogsReader:
    """
        A simple reader for VisDial v1.0 dialog data. The json file must have the
        same structure as mentioned on ``https://visualdialog.org/data``.

        Parameters
        ----------
        dialogs_json_path : str
            Path to json file containing VisDial v1.0 train, val or test data.
        num_examples: int, optional (default = None)
            Process first ``num_examples`` from the split. Useful to speed up while
            debugging.
        """

    def __init__(
            self,
            logger: AnaLogger,
            dialogs_json_path,
            num_examples=None,
    ):
        self.logger = logger
        if not os.path.exists(dialogs_json_path):
            self.logger.error('dialogs do not exist at %s' % dialogs_json_path)
            raise FileNotFoundError

        with open(dialogs_json_path, 'r') as visdial_file:
            visdial_data = json.load(visdial_file)
            # self.logger.debug(visdial_data)
        self._spilt = visdial_data['split']
        self.questions = {
            i: question
            for i, question in enumerate(visdial_data['data']['questions'])
        }
        self.answers = {
            i: answer
            for i, answer in enumerate(visdial_data['data']['answers'])
        }
        # Add empty question, answer - useful for padding dialog rounds.
        # for test split
        self.questions[-1], self.answers[-1] = '', ''

        # self.logger.debug(self.questions)

        # image_id serves as key for all four dicts here.
        self.captions: Dict[int, Any] = {}
        self.dialogs: Dict[int, Any] = {}
        self.num_rounds: Dict[int, Any] = {}
        self.original_indices: Dict[int, Any] = {}

        all_dialogs = visdial_data['data']['dialogs']

        # self.logger.debug(len(all_dialogs))

        # retain only first num_examples dialogs if specified.
        if num_examples is not None:
            all_dialogs = all_dialogs[:num_examples]
        for index, _dialog in enumerate(all_dialogs):
            self.captions[_dialog['image_id']] = _dialog['caption']
            self.original_indices[_dialog['image_id']] = index

            # record original length of dialog, before padding
            # 10 for train and val splits, 10 or less for test split
            self.num_rounds[_dialog['image_id']] = len(_dialog['dialog'])

            # pad dialog at the end with empty question and answer pairs
            # for test split
            while len(_dialog['dialog']) < 10:
                _dialog['dialog'].append({'question': -1, 'answer': -1})

            # add empty answer (and answer options) if not provided
            # for test split, use '-1' as a key for empty questions and answers
            for i in range(len(_dialog['dialog'])):
                if "answer" not in _dialog["dialog"][i]:
                    _dialog["dialog"][i]["answer"] = -1
                if "answer_options" not in _dialog["dialog"][i]:
                    _dialog["dialog"][i]["answer_options"] = [-1] * 100

            self.dialogs[_dialog['image_id']] = _dialog['dialog']

        # if num_examples is specified, collect questions and answers
        # included in those examples, and drop the rest to save time while
        # tokenizing. Collecting these should be fast because num_examples
        # during debugging are generally small.
        if num_examples is not None:
            questions_included: Set[int] = set()
            answers_included: Set[int] = set()

            for _dialog in self.dialogs.values():
                for _dialog_round in _dialog:
                    questions_included.add(_dialog_round['question'])
                    answers_included.add(_dialog_round['answer'])
                    for _answer_option in _dialog_round['answer_options']:
                        answers_included.add(_answer_option)
            self.questions = {
                i: self.questions[i]
                for i in questions_included
            }
            self.answers = {
                i: self.answers[i]
                for i in answers_included
            }

        self.logger.info(f'{self._spilt} tokenizing questions.')
        _question_tuples = self.questions.items()
        _question_indices, _question_contents = zip(*_question_tuples)
        _questions = list(tqdm(map(word_tokenize, _question_contents)))
        self.questions = {
            i: question + ['?']
            for i, question in zip(_question_indices, _questions)
        }
        # free memory
        del _question_tuples, _question_indices, _question_contents, _questions

        self.logger.info(f'{self._spilt} tokenizing answers.')
        _answer_tuples = self.answers.items()
        _answer_indices, _answer_contents = zip(*_answer_tuples)
        _answers = list(tqdm(map(word_tokenize, _answer_contents)))
        self.answers = {
            i: answer
            for i, answer in zip(_answer_indices, _answers)
        }
        del _answer_tuples, _answer_indices, _answer_contents, _answers

        self.logger.info(f'{self._spilt} tokenizing captions.')
        _caption_tuples = self.captions.items()
        _image_ids, _caption_contents = zip(*_caption_tuples)
        _captions = list(tqdm(map(word_tokenize, _caption_contents)))
        self.captions = {
            i: c
            for i, c in zip(_image_ids, _captions)
        }

    @property
    def spilt(self):
        return self._spilt

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, image_id) -> Dict[str, Any]:
        caption_for_image = self.captions[image_id]
        original_index = self.original_indices[image_id]
        num_rounds = self.num_rounds[image_id]
        dialog = copy.copy(self.dialogs[image_id])

        # replace question and answer indices with actual word tokens
        for i in range(len(dialog)):
            dialog[i]['question'] = self.questions[dialog[i]['question']]
            dialog[i]['answer'] = self.answers[dialog[i]['answer']]
            for answer_option_index, answer_option in enumerate(dialog[i]['answer_options']):
                dialog[i]['answer_options'][answer_option_index] = self.answers[answer_option]

        visdial_instance = {
            'image_id': image_id,
            'caption': caption_for_image,
            'dialog': dialog,
            'num_rounds': num_rounds,
        }

        return visdial_instance

    def keys(self):
        """
        :return: image_ids of dialogs
        """
        return list(self.dialogs.keys())


class DenseAnnotationsReader:
    """
        A reader for dense annotations for train/val split. The json file must have the
        same structure as mentioned on ``https://visualdialog.org/data``.

        Parameters
        ----------
        dense_annotations_json_path : str
            Path to a json file containing VisDial v1.0
        """

    def __init__(self, dense_annotations_json_path, logger: AnaLogger, split='train'):
        self.logger = logger
        self._split = split

        if not os.path.exists(dense_annotations_json_path):
            self.logger.error('dense annotations do not exist at %s' % dense_annotations_json_path)
            raise FileNotFoundError

        with open(dense_annotations_json_path, 'r') as visdial_file:
            self._visdial_data = json.load(visdial_file)
        self._image_ids = [
            entry['image_id'] for entry in self._visdial_data
        ]

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id) -> Dict[str, Union[int, list]]:
        index = self._image_ids.index(image_id)
        # keys: {'image_id', 'round_id', 'gt_relevance'}
        return self._visdial_data[index]

    # adding these APIs to get image indices for finetuning
    @property
    def all_data(self):
        return self._visdial_data

    @property
    def keys(self):
        return self._image_ids

    @property
    def split(self):
        return self._split


class ImageFeaturesReader:
    """
        A reader for HDF files containing pre-extracted image features. A typical
        HDF file is expected to have a column named "image_id", and another column
        named "features".

        Example of an HDF file:
        ```
        visdial_train_faster_rcnn_bottomup_features.h5
           |--- "image_id" [shape: (num_images, )]
           |--- "features" [shape: (num_images, num_proposals, feature_size)]
           +--- .attrs ("test", "train")
        ```
        Parameters
        ----------
        features_hdf_path : str
            Path to an HDF file containing VisDial v1.0 train, val or test split
            image features.
        in_memory : bool
            Whether to load the whole HDF file in memory. Beware, these files are
            sometimes tens of GBs in size. Set this to true if you have sufficient
            RAM - trade-off between speed and memory.
        """

    def __init__(self, features_hdf_path, logger: AnaLogger, in_memory=False):
        self.logger = logger
        self._in_memory = in_memory
        self.features_hdf_path = features_hdf_path

        if not os.path.exists(features_hdf_path):
            self.logger.error('image features do not exist at %s' % features_hdf_path)
            raise FileNotFoundError

        with h5py.File(features_hdf_path, 'r') as features_hdf:

            self._split = features_hdf.attrs['split']

            self._image_ids = list(map(int, features_hdf['image_id']))
            # 'features' is List[np.ndarray] if the dataset is loaded in-memory
            # if not loaded in memory, then list of None.
            self.features = [None] * len(self._image_ids)

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id):
        index = self._image_ids.index(image_id)
        if self._in_memory:
            # load features during first epoch, all not loaded together as it
            # has a slow start.
            if self.features[index] is not None:
                image_id_features = self.features[index]
            else:
                with h5py.File(self.features_hdf_path, 'r') as features_hdf:
                    image_id_features = features_hdf['features'][index]
                    self.features[index] = image_id_features
        else:
            # read chunk from file everytime if not loaded in memory.
            with h5py.File(self.features_hdf_path, 'r') as features_hdf:
                image_id_features = features_hdf['features'][index]
        return image_id_features

    def keys(self):
        return self._image_ids

    @property
    def spilt(self):
        return self._split


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    dr = DialogsReader(
        logger=logger,
        dialogs_json_path=opt.val_json,
        num_examples=5,
    )
    # logger.debug(dr[185565])
    dar = DenseAnnotationsReader(
        dense_annotations_json_path=opt.val_dense_json,
        logger=logger
    )
    # logger.debug(dar[445673])
    ifr = ImageFeaturesReader(
        features_hdf_path=opt.image_features_test_h5,
        logger=logger,
        in_memory=False
    )
    # logger.debug(ifr[295229].shape)
