# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abs_processor.py
# time: 2:53 下午

from abc import ABC

import numpy as np
from typing import Dict, List, Optional, Any

from kashgari.generators import CorpusGenerator
from kashgari.types import TextSamplesVar


class ABCProcessor(ABC):
    def info(self) -> Dict:
        return {
            'config': {
                'vocab2idx': self.vocab2idx,
            },
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__
        }

    def __init__(self, **kwargs: Any) -> None:
        self.vocab2idx = kwargs.get('vocab2idx', {})
        self.idx2vocab = dict([(v, k) for k, v in self.vocab2idx.items()])

        self.token_pad: str = kwargs.get('token_pad', '[PAD]')  # type: ignore
        self.token_unk: str = kwargs.get('token_unk', '[UNK]')  # type: ignore
        self.token_bos: str = kwargs.get('token_bos', '[BOS]')  # type: ignore
        self.token_eos: str = kwargs.get('token_eos', '[EOS]')  # type: ignore

    @property
    def vocab_size(self) -> int:
        return len(self.vocab2idx)

    @property
    def is_vocab_build(self) -> bool:
        return self.vocab_size != 0

    def build_vocab_dict_if_needs(self, generator: Optional[CorpusGenerator]) -> None:
        raise NotImplementedError

    def transform(self,
                  samples: TextSamplesVar,
                  *,
                  seq_length: int = None,
                  max_position: int = None,
                  segment: bool = False,
                  one_hot: bool = False,
                  **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self,
                          labels: List[int],
                          *,
                          lengths: List[int] = None,
                          **kwargs: Any) -> List[str]:
        raise NotImplementedError


if __name__ == "__main__":
    pass
