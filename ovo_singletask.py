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


x_dict = json_read("data/x_pro_dict.json")
y_dict = json_read("data/y_pro_dict.json")

names = list(x_dict.keys())
set_names = set(names)

x_dict = {name: np.array(data) for name, data in x_dict.items()}
y_dict = {name: np.array(data) for name, data in y_dict.items()}

n_layer = 10

for name in names:
    others = set_names.copy()
    others.remove(name)
    others = list(others)
    x_train = np.concatenate([x_dict[n] for n in others], axis=0)
    y_train = np.concatenate([y_dict[n] for n in others], axis=0)
    y_train = [y_train[:, i] for i in range(4)]

    x_test = x_dict[name]
    y_test = y_dict[name]
    y_test = [y_test[:, i] for i in range(4)]

    input_dim = x_train.shape[1]
    output_dim = [10, 48, 10, 48]
    output_name = ["TTtP", "ATaP", "TTtH", "ATaH"]

    for target in range(4):
        results = []
        if os.path.isfile(f"results/ovo_{name}_singletask_{target}.json"):
            results = json_read(f"results/ovo_{name}_singletask_{target}.json")

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
            for _ in range(10):
                m.set_weights(init)
                m.fit(
                    x=x_train,
                    y=y_train[target],
                    batch_size=64,
                    epochs=1000,
                    validation_split=0.2,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", mode="min", patience=10),
                        ModelCheckpoint(
                            filepath=f"models/ovo_{name}_singletask_{target}.h5",
                            monitor="val_loss",
                            mode="min",
                            save_best_only=True,
                        ),
                        # Metrics(),
                    ],
                )

                m.load_weights(f"models/ovo_{name}_singletask_{target}.h5")
                eval = m.evaluate(x=x_test, y=y_test[target])
                pprint.pprint(eval)
                results.append(eval)

        json_write(f"results/ovo_{name}_singletask_{target}.json", results)
