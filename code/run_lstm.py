import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, LSTM, GRU, Bidirectional
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="0"
abs_path=os.path.abspath('..')

with open(abs_path + "/dataset/sigodd.txt", "rb") as fp:  # Unpickling
    sigodd = np.array(pickle.load(fp))
with open(abs_path + "/dataset/signodd.txt", "rb") as fp:  # Unpickling
    signodd = np.array(pickle.load(fp))

"""
with open(abs_path + "/dataset/twoodd.txt", "rb") as fp:  # Unpickling
    twoodd = np.array(pickle.load(fp))
with open(abs_path + "/dataset/twonodd.txt", "rb") as fp:  # Unpickling
    twonodd = np.array(pickle.load(fp))


sigodd = sigodd.reshape(15488,2049)
signodd = signodd.reshape(54912,2049)
twoodd = twoodd.reshape(13904,2049)
twonodd = twonodd.reshape(56496,2049)
"""



train_odd_label = np.ones(88)
train_nodd_label = np.zeros(88)


train_data_set = np.concatenate((sigodd, signodd[0:88]))

train_label_set = np.concatenate((train_odd_label, train_nodd_label))


np.random.seed(7)
np.random.shuffle(train_data_set)
np.random.seed(7)
np.random.shuffle(train_label_set)
tf.random.set_seed(7)

train_data_set = np.reshape(train_data_set, (len(train_data_set), 176, 2049))



model = tf.keras.Sequential([
    Bidirectional(LSTM(256, use_bias = True,return_sequences = True)),
    Dropout(0.5),
    Bidirectional(LSTM(32, use_bias = True)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1),
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])


history = model.fit(train_data_set, train_label_set, batch_size=32, epochs=50, validation_split=0.2, validation_freq=1,)


print(history.history)
# 显示训练集和验证集的acc和loss曲线

loss = history.history['accuracy']
val_loss = history.history['val_accuracy']

plt.plot(loss, label='Training Accuracy')
plt.plot(val_loss, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

model.summary()