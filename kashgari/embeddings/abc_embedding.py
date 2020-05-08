# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abc_embedding.py
# time: 2:43 下午

import json
import logging
import pydoc

import numpy as np
import tqdm
from tensorflow import keras
from typing import Dict, List, Any

import kashgari
from kashgari.generators import CorpusGenerator
from kashgari.processors import SequenceProcessor
from kashgari.processors.abc_processor import ABCProcessor
from kashgari.types import TextSamplesVar, LabelSamplesVar

L = keras.layers


class ABCEmbedding:
    def info(self) -> Dict:
        config: Dict[str, Any] = {
            'sequence_length': self.sequence_length,
            'segment': self.segment,
            'max_position': self.max_position,
            **self.kwargs
        }
        return {
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'text_processor': self.text_processor.info(),
            'label_processor': self.label_processor.info(),
            'config': config,
            'embed_model': json.loads(self.embed_model.to_json())
        }

    @classmethod
    def load_saved_model_embedding(cls,
                                   config_dict: Dict,
                                   **kwargs: Any) -> 'ABCEmbedding':

        target_name = 'text_processor'
        module_name = f"{config_dict[target_name]['module']}.{config_dict[target_name]['class_name']}"
        text_processor = pydoc.locate(module_name)(**config_dict[target_name]['config'])  # type: ignore

        target_name = 'label_processor'
        module_name = f"{config_dict[target_name]['module']}.{config_dict[target_name]['class_name']}"
        label_processor = pydoc.locate(module_name)(**config_dict[target_name]['config'])  # type: ignore

        instance = cls(text_processor=text_processor,
                       label_processor=label_processor,
                       **config_dict['config'])

        embed_model_json_str = json.dumps(config_dict['embed_model'])
        instance.embed_model = keras.models.model_from_json(embed_model_json_str,
                                                            custom_objects=kashgari.custom_objects)
        return instance

    def __init__(self,
                 sequence_length: int = None,
                 text_processor: ABCProcessor = None,
                 label_processor: ABCProcessor = None,
                 **kwargs: Any):

        if text_processor is None:
            text_processor = SequenceProcessor()

        self.text_processor: ABCProcessor = text_processor
        self.label_processor: ABCProcessor = label_processor  # type: ignore
        self.embed_model: keras.Model = None

        self.sequence_length = sequence_length
        self.segment: bool = kwargs.get('segment', False)  # type: ignore
        self.kwargs = kwargs

        self.embedding_size: int = kwargs.get('embedding_size', None)  # type: ignore
        self.max_position: int = kwargs.get('max_position', None)  # type: ignore

    def set_sequence_length(self, length: int) -> None:
        self.sequence_length = length
        if self.embed_model is not None:
            logging.info(f"Rebuild embedding model with sequence length: {length}")
            self.embed_model = None
            self.build_embedding_model()

    def calculate_sequence_length_if_needs(self,
                                           corpus_gen: CorpusGenerator,
                                           *,
                                           cover_rate: float = 0.95) -> None:
        if self.sequence_length is None:
            seq_lens = []
            for sentence, _ in tqdm.tqdm(corpus_gen,
                                         desc="Calculating sequence length"):
                seq_lens.append(len(sentence))
            self.sequence_length = sorted(seq_lens)[int(cover_rate * len(seq_lens))]
            logging.warning(f'Calculated sequence length = {self.sequence_length}')

    def build(self, x_data: TextSamplesVar, y_data: LabelSamplesVar) -> None:
        gen = CorpusGenerator(x_data=x_data, y_data=y_data)
        self.build_with_generator(gen)

    def build_with_generator(self, gen: CorpusGenerator = None) -> None:
        self.build_text_vocab(gen=gen)
        self.build_embedding_model()
        if self.label_processor and gen is not None:
            self.label_processor.build_vocab_dict_if_needs(gen)

    def build_text_vocab(self, gen: CorpusGenerator = None, *, force: bool = False) -> None:
        raise NotImplementedError

    def build_embedding_model(self) -> None:
        raise NotImplementedError

    def embed(self,
              sentences: List[List[str]],
              debug: bool = False) -> np.ndarray:
        """
        batch embed sentences

        Args:
            sentences: Sentence list to embed
            debug: show debug info
        Returns:
            vectorized sentence list
        """
        self.build_with_generator()
        tensor_x = self.text_processor.transform(sentences,
                                                 segment=self.segment,
                                                 seq_length=self.sequence_length)
        if debug:
            logging.debug(f'sentence tensor: {tensor_x}')
        embed_results = self.embed_model.predict(tensor_x)
        return embed_results


if __name__ == "__main__":
    pass
