# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: utils.py
# time: 12:39 下午

import os
import json
import pydoc
import random
import numpy as np
from typing import List, Union, TypeVar, Tuple, Type, Dict

from tensorflow import keras

from kashgari import custom_objects
from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.tasks.classification.abc_model import ABCClassificationModel
from kashgari.tasks.labeling.abc_model import ABCLabelingModel
from kashgari.types import MultiLabelClassificationLabelVar

T = TypeVar("T")


def get_list_subset(target: List[T], index_list: List[int]) -> List[T]:
    """
    Get the subset of the target list
    Args:
        target: target list
        index_list: subset items index

    Returns:
        subset of the original list
    """
    return [target[i] for i in index_list if i < len(target)]


def unison_shuffled_copies(a: List[T],
                           b: List[T]) -> Union[Tuple[List[T], ...], Tuple[np.ndarray, ...]]:
    """
    Union shuffle two arrays
    Args:
        a:
        b:

    Returns:

    """
    data_type = type(a)
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    if data_type == np.ndarray:
        return np.array(a), np.array(b)
    return list(a), list(b)


def custom_object_scope():
    return keras.utils.custom_object_scope(custom_objects)


def load_model(model_path: str, load_weights: bool = True) -> Union[ABCClassificationModel, ABCLabelingModel]:
    """
    Load saved model from saved model from `model.save` function
    Args:
        model_path: model folder path
        load_weights: only load model structure and vocabulary when set to False, default True.

    Returns:
        Loaded kashgari model
    """
    from bert4keras.layers import ConditionalRandomField
    with open(os.path.join(model_path, 'model_info.json'), 'r') as f:
        model_info = json.load(f)

    model_class: Type[ABCClassificationModel] = pydoc.locate(f"{model_info['module']}.{model_info['class_name']}")
    model_json_str = json.dumps(model_info['tf_model'])

    embed_info = model_info['embedding']
    embed_class: Type[ABCEmbedding] = pydoc.locate(f"{embed_info['module']}.{embed_info['class_name']}")
    embedding: ABCEmbedding = embed_class.load_saved_model_embedding(embed_info)

    model = model_class(embedding=embedding)
    model.tf_model = keras.models.model_from_json(model_json_str, custom_objects)
    if load_weights:
        model.tf_model.load_weights(os.path.join(model_path, 'model_weights.h5'))

    # Load Weights from model
    for layer in embedding.embed_model.layers:
        layer.set_weights(model.tf_model.get_layer(layer.name).get_weights())

    if isinstance(model.tf_model.layers[-1], ConditionalRandomField):
        model.layer_crf = model.tf_model.layers[-1]

    return model


class MultiLabelBinarizer:
    def __init__(self, vocab2idx: Dict[str, int]):
        self.vocab2idx = vocab2idx
        self.idx2vocab = dict([(v, k) for k, v in vocab2idx.items()])

    @property
    def classes(self) -> List[str]:
        return list(self.idx2vocab.values())

    def transform(self, samples: MultiLabelClassificationLabelVar) -> np.ndarray:
        data = np.zeros((len(samples), len(self.vocab2idx)))
        for sample_index, sample in enumerate(samples):
            for label in sample:
                data[sample_index][self.vocab2idx[label]] = 1
        return data

    def inverse_transform(self, preds: np.ndarray, threshold: float = 0.5) -> List[List[str]]:
        data = []
        for sample in preds:
            x = []
            for label_x in np.where(sample >= threshold)[0]:
                x.append(self.idx2vocab[label_x])
            data.append(x)
        return data


if __name__ == "__main__":
    a_b = MultiLabelBinarizer(vocab2idx={'identity_hate': 0,
                                         'insult': 1,
                                         'obscene': 2,
                                         'severe_toxic': 3,
                                         'threat': 4,
                                         'toxic': 5}
                              )
    r = a_b.transform([[], ['identity_hate'], ['obscene', 'threat']])
    print(r)
    xx = np.random.random((10, 6))
    print(a_b.inverse_transform(xx, threshold=0.9))
