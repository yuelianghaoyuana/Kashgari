# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: label_processor.py
# time: 2:53 下午

import collections
import operator

import numpy as np
import tqdm
from tensorflow.keras.utils import to_categorical
from typing import List, Union, Dict

from kashgari.types import TextSamplesVar, ClassificationLabelVar, MultiLabelClassificationLabelVar
from kashgari.generators import CorpusGenerator
from kashgari.processors.abc_processor import ABCProcessor


class ClassificationProcessor(ABCProcessor):

    def info(self) -> Dict:
        data = super(ClassificationProcessor, self).info()
        data['config']['multi_label'] = self.multi_label
        return data

    def __init__(self,
                 multi_label: bool = False,
                 **kwargs: Dict) -> None:
        from kashgari.utils import MultiLabelBinarizer
        super(ClassificationProcessor, self).__init__(**kwargs)
        self.multi_label = multi_label
        self.multi_label_binarizer = MultiLabelBinarizer(self.vocab2idx)

    def build_vocab_dict_if_needs(self, generator: CorpusGenerator) -> None:
        from kashgari.utils import MultiLabelBinarizer
        if self.vocab2idx:
            return

        vocab2idx: Dict[str, int] = {}
        token2count: Dict[str, int] = {}

        if self.multi_label:
            for _, label in tqdm.tqdm(generator, desc="Preparing classification label vocab dict"):
                for token in label:
                    count = token2count.get(token, 0)
                    token2count[token] = count + 1
        else:
            for _, label in tqdm.tqdm(generator, desc="Preparing classification label vocab dict"):
                count = token2count.get(label, 0)
                token2count[label] = count + 1

        sorted_token2count = sorted(token2count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)
        token2count = collections.OrderedDict(sorted_token2count)

        for token, token_count in token2count.items():
            if token not in vocab2idx:
                vocab2idx[token] = len(vocab2idx)
        self.vocab2idx = vocab2idx
        self.idx2vocab = dict([(v, k) for k, v in self.vocab2idx.items()])
        self.multi_label_binarizer = MultiLabelBinarizer(self.vocab2idx)

    def transform(self,
                  samples: TextSamplesVar,
                  *,
                  seq_length: int = None,
                  max_position: int = None,
                  segment: bool = False,
                  one_hot: bool = False,
                  **kwargs: Dict) -> np.ndarray:
        if self.multi_label:
            sample_tensor = self.multi_label_binarizer.transform(samples)
            return sample_tensor

        sample_tensor = [self.vocab2idx[i] for i in samples]
        if one_hot:
            return to_categorical(sample_tensor, self.vocab_size)
        else:
            return np.array(sample_tensor)

    def inverse_transform(self,
                          labels: Union[List[int], np.ndarray],
                          *,
                          lengths: List[int] = None,
                          **kwargs: Dict) -> List[str]:
        return [self.idx2vocab[i] for i in labels]


if __name__ == "__main__":
    pass
