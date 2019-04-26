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

input_dim = x.shape[1]
output_dim = [10, 48, 10, 48]
output_name = ["TTtP", "ATaP", "TTtH", "ATaH"]
n_layer = 10


for target in range(4):
    results = []
    if os.path.isfile(f"results/singletask_{target}.json"):
        results = json_read(f"results/singletask_{target}.json")

    for l in range(1, n_layer):
        h_units = pyramid(input_dim, output_dim[target], l)

        i = Input(shape=[input_dim])
        h = i
        for units in h_units:
            h = Dense(units, activation="relu")(h)
            h = Dropout(0.5)(h)
        o = Dense(output_dim[target], activation="softmax", name=output_name[0])(h)

        m = Model(inputs=i, outputs=o)
        m.compile("adam", "sparse_categorical_crossentropy", metrics=["acc"])

        init = m.get_weights()
        for _ in range(50):
            m.set_weights(init)
            m.fit(
                x=x_train,
                y=y_train[target],
                batch_size=64,
                epochs=1000,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min", patience=20),
                    ModelCheckpoint(
                        filepath=f"models/singletask_{target}.h5",
                        monitor="val_loss",
                        mode="min",
                        save_best_only=True,
                    ),
                    # Metrics(),
                ],
            )

            m.load_weights(f"models/singletask_{target}.h5")
            eval = m.evaluate(x=x_test, y=y_test[target])
            pprint.pprint(eval)
            results.append(eval)

    json_write(f"results/singletask_{target}.json", results)
