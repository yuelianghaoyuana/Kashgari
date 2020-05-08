# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: __init__.py
# time: 1:57 下午

import os
import time
import unittest
import tempfile

from tests.test_macros import TestMacros

from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import WordEmbedding
from kashgari.tasks.classification import BiLSTM_Model
from kashgari.utils import load_model


class TestBiLSTM_Model(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = BiLSTM_Model
        cls.w2v_embedding = WordEmbedding(TestMacros.w2v_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.w2v_embedding = None

    def test_basic_use(self):
        model = self.TASK_MODEL_CLASS(sequence_length=20)
        train_x, train_y = SMP2018ECDTCorpus.load_data()
        valid_x, valid_y = train_x, train_y

        model.fit(train_x,
                  train_y,
                  x_validate=valid_x,
                  y_validate=valid_y,
                  epochs=self.EPOCH_COUNT)

        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        original_y = model.predict(train_x[:20])
        model.save(model_path)
        del model
        new_model = load_model(model_path)
        new_model.tf_model.summary()
        new_y = new_model.predict(train_x[:20])
        assert new_y == original_y

        new_model.evaluate(valid_x, valid_y)

    def test_multi_label(self):
        corpus = TestMacros.jigsaw_mini_corpus
        model = self.TASK_MODEL_CLASS(sequence_length=20, multi_label=True)
        x, y = corpus.load_data()
        model.fit(x, y, epochs=self.EPOCH_COUNT)

        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        original_y = model.predict(x[:20])
        model.save(model_path)
        del model
        new_model = load_model(model_path)
        new_model.tf_model.summary()
        new_y = new_model.predict(x[:20])

        assert new_y == original_y

        new_model.evaluate(x, y)

    def test_with_word_embedding(self):
        self.w2v_embedding.set_sequence_length(50)
        model = self.TASK_MODEL_CLASS(embedding=self.w2v_embedding)
        train_x, train_y = SMP2018ECDTCorpus.load_data()
        valid_x, valid_y = train_x, train_y

        model.fit(train_x,
                  train_y,
                  x_validate=valid_x,
                  y_validate=valid_y,
                  epochs=self.EPOCH_COUNT)

        with self.assertRaises(ValueError):
            model = self.TASK_MODEL_CLASS(embedding=self.w2v_embedding,
                                          sequence_length=100)


if __name__ == '__main__':
    unittest.main()
