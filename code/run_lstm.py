import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, LSTM, GRU, Bidirectional
import matplotlib.pyplot as plt

# abs_path=os.path.abspath('..')

# with open("D:\dataset\SEEG-Oddball/dataset/sigodd.txt", "rb") as fp:  # Unpickling
#     sigodd = np.array(pickle.load(fp))
# with open("D:\dataset\SEEG-oddball/dataset/signodd.txt", "rb") as fp:  # Unpickling
#     signodd = np.array(pickle.load(fp))

# train_odd_label = np.ones(88)
# train_nodd_label = np.zeros(88)
#
# train_data_set = np.concatenate((sigodd, signodd[0:88]))
# train_label_set = np.concatenate((train_odd_label, train_nodd_label))
#
# np.save('D:\dataset\SEEG-Oddball/dataset/seeg_data.npy', train_data_set)
# np.save('D:\dataset\SEEG-Oddball/dataset/seeg_label.npy', train_label_set)

train_data_set = np.load('D:\dataset\SEEG-Oddball/dataset/seeg_data.npy')
train_label_set = np.load('D:\dataset\SEEG-Oddball/dataset/seeg_label.npy')

n_seed = 8
np.random.seed(n_seed)
np.random.shuffle(train_data_set)
np.random.seed(n_seed)
np.random.shuffle(train_label_set)
# tf.random.set_seed(n_seed)

train_data_set = np.reshape(train_data_set, (len(train_data_set), 176, 2049))


model = tf.keras.Sequential([
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1),
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])


history = model.fit(train_data_set, train_label_set, batch_size=32, epochs=50, validation_split=0.2, validation_freq=1)

# 显示训练集和验证集的acc曲线
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

model.summary()