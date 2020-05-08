# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abs_model.py
# time: 4:05 下午

from abc import ABC
from typing import List, Dict, Any, Tuple, Union, TYPE_CHECKING

import random
import numpy as np
from sklearn import metrics

import kashgari
from kashgari.logger import logger
from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.types import TextSamplesVar, ClassificationLabelVar, MultiLabelClassificationLabelVar
from kashgari.generators import CorpusGenerator
from kashgari.tasks.abs_task_model import ABCTaskModel
from kashgari.processors.class_processor import ClassificationProcessor
from kashgari.generators import BatchDataGenerator
from kashgari.layers import L

if TYPE_CHECKING:
    from tensorflow import keras


class ABCClassificationModel(ABCTaskModel, ABC):
    __task__ = 'classification'

    def info(self) -> Dict:
        info = super(ABCClassificationModel, self).info()
        info['config']['multi_label'] = self.multi_label
        return info

    def __init__(self,
                 embedding: ABCEmbedding = None,
                 *,
                 sequence_length: int = None,
                 hyper_parameters: Dict[str, Dict[str, Any]] = None,
                 multi_label: bool = False,
                 **kwargs: Dict):
        """
        Abstract Classification Model
        Args:
            embedding: embedding object
            sequence_length: target sequence length
            hyper_parameters: hyper_parameters to overwrite
            **kwargs:
        """
        super(ABCClassificationModel, self).__init__(embedding=embedding,
                                                     sequence_length=sequence_length,
                                                     hyper_parameters=hyper_parameters,
                                                     **kwargs)
        self.multi_label = multi_label
        self.default_labeling_processor = ClassificationProcessor(multi_label=self.multi_label)

    def _activation_layer(self) -> L.Layer:
        if self.multi_label:
            return L.Activation('sigmoid')
        else:
            return L.Activation('softmax')

    def compile_model(self, **kwargs: Any) -> None:
        if kwargs.get('loss') is None:
            if self.multi_label:
                kwargs['loss'] = 'binary_crossentropy'
            else:
                kwargs['loss'] = 'categorical_crossentropy'

        super(ABCClassificationModel, self).compile_model(**kwargs)

    def fit(self,
            x_train: TextSamplesVar,
            y_train: Union[ClassificationLabelVar, MultiLabelClassificationLabelVar],
            x_validate: TextSamplesVar = None,
            y_validate: Union[ClassificationLabelVar, MultiLabelClassificationLabelVar] = None,
            *,
            batch_size: int = 64,
            epochs: int = 5,
            callbacks: List['keras.callbacks.Callback'] = None,
            fit_kwargs: Dict = None) -> 'keras.callbacks.History':
        """
        Trains the model for a given number of epochs with given data set list.

        Args:
            x_train: Array of train feature data (if the model has a single input),
                or tuple of train feature data array (if the model has multiple inputs)
            y_train: Array of train label data
            x_validate: Array of validation feature data (if the model has a single input),
                or tuple of validation feature data array (if the model has multiple inputs)
            y_validate: Array of validation label data
            batch_size: Number of samples per gradient update, default to 64.
            epochs: Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y` data provided.
                Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`.
            fit_kwargs: fit_kwargs: additional arguments passed to ``fit()`` function from
                ``tensorflow.keras.Model`` - https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        train_gen = CorpusGenerator(x_train, y_train)
        if x_validate is not None:
            valid_gen = CorpusGenerator(x_validate, y_validate)
        else:
            valid_gen = None
        return self.fit_generator(train_sample_gen=train_gen,
                                  valid_sample_gen=valid_gen,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  fit_kwargs=fit_kwargs)

    def fit_generator(self,
                      train_sample_gen: CorpusGenerator,
                      valid_sample_gen: CorpusGenerator = None,
                      *,
                      batch_size: int = 64,
                      epochs: int = 5,
                      callbacks: List['keras.callbacks.Callback'] = None,
                      fit_kwargs: Dict = None) -> 'keras.callbacks.History':
        """
        Trains the model for a given number of epochs with given data generator.

        Data generator must be the subclass of `CorpusGenerator`

        Args:
            train_sample_gen: train data generator.
            valid_sample_gen: valid data generator.
            batch_size: Number of samples per gradient update, default to 64.
            epochs: Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y` data provided.
                Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`.
            fit_kwargs: fit_kwargs: additional arguments passed to ``fit()`` function from
                ``tensorflow.keras.Model`` - https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        self.build_model(train_sample_gen)
        self.tf_model.summary()

        train_gen = BatchDataGenerator(train_sample_gen,
                                       text_processor=self.embedding.text_processor,
                                       label_processor=self.embedding.label_processor,
                                       segment=self.embedding.segment,
                                       seq_length=self.embedding.sequence_length,
                                       batch_size=batch_size)

        if fit_kwargs is None:
            fit_kwargs = {}

        if valid_sample_gen:
            valid_gen = BatchDataGenerator(valid_sample_gen,
                                           text_processor=self.embedding.text_processor,
                                           label_processor=self.embedding.label_processor,
                                           segment=self.embedding.segment,
                                           seq_length=self.embedding.sequence_length,
                                           batch_size=batch_size)
            fit_kwargs['validation_data'] = valid_gen.generator()
            fit_kwargs['validation_steps'] = len(valid_gen)

        return self.tf_model.fit(train_gen.generator(),
                                 steps_per_epoch=len(train_gen),
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 **fit_kwargs)

    def predict(self,  # type: ignore[override]
                x_data: TextSamplesVar,
                *,
                batch_size: int = 32,
                truncating: bool = False,
                multi_label_threshold: float = 0.5,
                debug_info: bool = False,
                predict_kwargs: Dict = None,
                **kwargs: Any) -> Union[ClassificationLabelVar, MultiLabelClassificationLabelVar]:
        """
        Generates output predictions for the input samples.

        Computation is done in batches.

        Args:
            x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            truncating: remove values from sequences larger than `model.embedding.sequence_length`
            multi_label_threshold:
            debug_info: Bool, Should print out the logger info.
            predict_kwargs: arguments passed to ``predict()`` function of ``tf.keras.Model``

        Returns:
            array(s) of predictions.
        """
        if predict_kwargs is None:
            predict_kwargs = {}
        with kashgari.utils.custom_object_scope():
            if truncating:
                seq_length = self.embedding.sequence_length
            else:
                seq_length = None
            tensor = self.text_processor.transform(x_data,
                                                   segment=self.embedding.segment,
                                                   seq_lengtg=seq_length,
                                                   max_position=self.embedding.max_position)
            pred = self.tf_model.predict(tensor, batch_size=batch_size, **predict_kwargs)

            if self.multi_label:
                if debug_info:
                    print('raw output: {}'.format(pred))
                multi_label_binarizer = self.label_processor.multi_label_binarizer  # type: ignore
                res = multi_label_binarizer.inverse_transform(pred,
                                                              threshold=multi_label_threshold)
            else:
                pred = pred.argmax(-1)
                lengths = [len(sen) for sen in x_data]
                res = self.embedding.label_processor.inverse_transform(pred,
                                                                       lengths=lengths)

            if debug_info:
                print('input: {}'.format(tensor))
                print('output: {}'.format(pred))
                print('output argmax: {}'.format(pred.argmax(-1)))
        return res

    def evaluate(self,
                 x_data: TextSamplesVar,
                 y_data: Union[ClassificationLabelVar, MultiLabelClassificationLabelVar],
                 *,
                 batch_size: int = 32,
                 digits: int = 4,
                 multi_label_threshold: float = 0.5,
                 truncating: bool = False,
                 debug_info: bool = False) -> Dict:
        y_pred = self.predict(x_data,
                              batch_size=batch_size,
                              truncating=truncating,
                              multi_label_threshold=multi_label_threshold,
                              debug_info=debug_info)

        if debug_info:
            for index in random.sample(list(range(len(x_data))), 5):
                logger.debug('------ sample {} ------'.format(index))
                logger.debug('x      : {}'.format(x_data[index]))
                logger.debug('y      : {}'.format(y_data[index]))
                logger.debug('y_pred : {}'.format(y_pred[index]))

        if self.multi_label:
            multi_label_binarizer = self.label_processor.multi_label_binarizer  # type: ignore
            y_pred_b = multi_label_binarizer.transform(y_pred)
            y_true_b = multi_label_binarizer.transform(y_data)

            # hamming_loss = metrics.hamming_loss(y_pred_b, y_true_b)
            report = {}
            for c_index, c in enumerate(multi_label_binarizer.classes):
                precision = metrics.precision_score(y_true_b[:, c_index], y_pred_b[:, c_index])
                recall = metrics.recall_score(y_true_b[:, c_index], y_pred_b[:, c_index])
                f1 = metrics.f1_score(y_true_b[:, c_index], y_pred_b[:, c_index])
                support = len(np.where(y_true_b[:, c_index] == 1)[0])
                report[c] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': support
                }

            headers = ["precision", "recall", "f1-score", "support"]
            head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
            print(head_fmt.format('', *headers, width=20))

            rows = []
            row_fmt = '{:>{width}s}  {:>9.{digits}f} {:>9.{digits}f} {:>9.{digits}f} {:>9}\n'

            for k, v in report.items():
                rows.append((k, v['precision'], v['recall'], v['f1'], v['support']))

            for row in rows:
                print(row_fmt.format(*row, width=20, digits=4))

        else:
            report = metrics.classification_report(y_data,
                                                   y_pred,
                                                   output_dict=True,
                                                   digits=digits)
            print(metrics.classification_report(y_data,
                                                y_pred,
                                                output_dict=False,
                                                digits=digits))
        # TODO: Fix report
        return report


if __name__ == "__main__":
    pass
