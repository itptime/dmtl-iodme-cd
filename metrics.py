import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support
from keras.backend import argmax


class Metrics(Callback):
    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = self.model.predict(self.validation_data[0])
        if isinstance(val_predict, list):
            for target in range(len(val_predict)):
                val_targ = self.validation_data[1 + target].ravel()
                val_pred = np.argmax(val_predict[target], axis=-1)

                _val_precision, _val_recall, _val_f1, _ = precision_recall_fscore_support(
                    val_targ, val_pred, average="micro"
                )
                self.val_f1s.append(_val_f1)
                self.val_recalls.append(_val_recall)
                self.val_precisions.append(_val_precision)
                print(
                    f" - val_f1_{target}: {_val_f1}"
                    + f" - val_precision_{target}: {_val_precision}"
                    + f" - val_recall_{target} {_val_recall}"
                )
        else:
            val_targ = self.validation_data[1].ravel()
            val_pred = np.argmax(val_predict, axis=-1)

            _val_precision, _val_recall, _val_f1, _ = precision_recall_fscore_support(
                val_targ, val_pred, average="micro"
            )
            self.val_f1s.append(_val_f1)
            self.val_recalls.append(_val_recall)
            self.val_precisions.append(_val_precision)
            print(
                " - val_f1: %f - val_precision: %f - val_recall %f"
                % (_val_f1, _val_precision, _val_recall)
            )
