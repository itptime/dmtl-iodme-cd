import os
import pprint

import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, Input, concatenate, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from io_json import json_read, json_write
from metrics import Metrics
from utils import pyramid


x = np.array(json_read("data/x_pro.json"))
y = np.array(json_read("data/y_pro.json"))


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
y_train = [y_train[:, i] for i in range(4)]
y_test = [y_test[:, i] for i in range(4)]

_run = 2
n_layer = 10
input_dim = x.shape[1]
output_dim = [10, 48, 10, 48]
output_name = ["TTtP", "ATaP", "TTtH", "ATaH"]
output_loss = [0.5, 1.0, 0.5, 1.0]

results = []
if os.path.isfile(f"results/multitask_{_run}.json"):
    results = json_read(f"results/multitask_{_run}.json")

for l in range(1, n_layer):
    h_units = pyramid(input_dim, sum(output_dim), l)

    i = Input(shape=[input_dim])
    h = i
    for units in h_units:
        h = Dense(units, activation="relu")(h)
        h = Dropout(0.5)(h)
    o = [
        Dense(dim, activation="softmax", name=name)(h)
        for dim, name in zip(output_dim, output_name)
    ]

    m = Model(inputs=i, outputs=o)
    m.compile(
        "adam",
        "sparse_categorical_crossentropy",
        metrics=["accuracy"],
        loss_weights=output_loss,
    )


    init = m.get_weights()
    for _ in range(50):
        m.set_weights(init)
        m.fit(
            x=x_train,
            y=y_train,
            batch_size=64,
            epochs=1000,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=20),
                ModelCheckpoint(
                    filepath=f"models/multitask_model_{_run}.h5",
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                ),
                # Metrics(),
            ],
        )

        m.load_weights(f"models/multitask_model_{_run}.h5")
        eval = m.evaluate(x=x_test, y=y_test)
        # y_pred = [np.argmax(y_p) for y_p in m.predict(x=x_test)]
        # y_equal = [(y_t == y_p)[:, np.newaxis] for y_t, y_p in zip(y_test, y_pred)]
        # y_equal = np.all(np.concatenate([y_equal[1], y_equal[3]], axis=1), axis=1)
        # all_accuracy = np.sum(y_equal) / y_equal.shape[0]
        # eval = eval[:-4] + [all_accuracy] + eval[-4:]
        pprint.pprint(eval)
        results.append(eval)

json_write(f"results/multitask_{_run}.json", results)
