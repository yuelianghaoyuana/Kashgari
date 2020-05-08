# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: eval_callBack.py
# time: 6:53 下午

from tensorflow import keras

from kashgari.tasks.abs_task_model import ABCTaskModel
from typing import List, Any, Dict


class EvalCallBack(keras.callbacks.Callback):

    def __init__(self,
                 *,
                 task_model: ABCTaskModel,
                 valid_x: List[Any],
                 valid_y: List[Any],
                 step: int = 5,
                 batch_size: int = 256) -> None:
        """
        Evaluate callback, calculate precision, recall and f1
        Args:
            task_model: the kashgari task model to evaluate
            valid_x: feature data
            valid_y: label data
            step: step, default 5
            batch_size: batch size, default 256
        """
        super(EvalCallBack, self).__init__()
        self.task_model: ABCTaskModel = task_model
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.step = step
        self.batch_size = batch_size
        self.logs: List[Dict] = []

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        if (epoch + 1) % self.step == 0:
            report = self.task_model.evaluate(self.valid_x,
                                              self.valid_y,
                                              batch_size=self.batch_size)

            self.logs.append({
                'precision': report['precision'],
                'recall': report['recall'],
                'f1-score': report['f1-score']
            })
            print(f"\nepoch: {epoch} precision: {report['precision']:.6f},"
                  f" recall: {report['recall']:.6f}, f1-score: {report['f1-score']:.6f}")
