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
from typing import Dict
from tensorflow.keras.utils import to_categorical
from typing import List

from kashgari.generators import CorpusGenerator
from kashgari.processors.abc_processor import ABCProcessor


class ClassificationProcessor(ABCProcessor):

    def info(self) -> Dict:
        data = super(ClassificationProcessor, self).info()
        data['config']['multi_label'] = self.multi_label
        return data

    def __init__(self,
                 multi_label: bool = False,
                 **kwargs):
        from kashgari.utils import MultiLabelBinarizer
        super(ClassificationProcessor, self).__init__(**kwargs)
        self.multi_label = multi_label
        self.multi_label_binarizer = MultiLabelBinarizer(self.vocab2idx)

    def build_vocab_dict_if_needs(self, generator: CorpusGenerator):
        from kashgari.utils import MultiLabelBinarizer
        if self.vocab2idx:
            return
        vocab2idx = {}

        token2count = {}

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

    def numerize_samples(self,
                         samples: List[str],
                         seq_length: int = None,
                         one_hot: bool = False,
                         **kwargs) -> np.ndarray:
        if self.multi_label:
            sample_tensor = self.multi_label_binarizer.transform(samples)
            return sample_tensor

        sample_tensor = [self.vocab2idx[i] for i in samples]
        if one_hot:
            return to_categorical(sample_tensor, self.vocab_size)
        else:
            return np.array(sample_tensor)

    def reverse_numerize(self,
                         indexs: List[str],
                         lengths: List[int] = None,
                         **kwargs) -> List[str]:
        return [self.idx2vocab[i] for i in indexs]


if __name__ == "__main__":
    pass
