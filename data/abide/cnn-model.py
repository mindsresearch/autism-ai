import os

import tensorflow as tf
import tensorflow.keras.layers as tfl
import nibabel as nib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd

NIIROOT = 'nyu_niigz/'
CSVPATH = 'subject_data.csv'
niipaths = []
for root, dirs, files in os.walk(NIIROOT):
    for file in files:
        if file.endswith('.gz'):
            niipaths.append((root, file))

data = [(p[1][:-12], nib.load(os.path.join(p[0], p[1])).get_fdata()) for p in tqdm(niipaths, desc='load')]
df = pd.read_csv(CSVPATH)

y = np.array([[list(df[df['FILE_ID'] == x[0]]['DX_GROUP'])[0]-1 for x in tqdm(data, desc='pain')]]).T

X = [x[1] for x in data]

model = tf.keras.Sequential(name='abide_cnn')

X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.4, random_state=202410350)
X_dv, X_te, y_dv, y_te = train_test_split(X_vl, y_vl, random_state=350202410)

X_tr = tf.constant(np.array(X_tr))
y_tr = tf.constant(np.array(y_tr))
X_dv = tf.constant(np.array(X_dv))
y_dv = tf.constant(np.array(y_dv))
X_te = tf.constant(np.array(X_te))
y_te = tf.constant(np.array(y_te))

# print("y_tr:")
# print(y_tr)

# print('y_dv:')
# print(y_dv)

print(f"--------\nDATA SHAPES:\n  tr: {X_tr.shape}\n  dv: {X_dv.shape}\n  te: {X_te.shape}\n--------")

# Model architecture copied from Alharthi & Alzahrani (11/2023)
# DOI: 10.3390/brainsci13111578

model.add(tfl.InputLayer(input_shape=(61, 73, 61, 1)))

model.add(tfl.Conv3D(64, (3, 3, 3), activation='relu'))
model.add(tfl.MaxPooling3D((2, 2, 2)))
model.add(tfl.BatchNormalization())

model.add(tfl.Conv3D(64, (3, 3, 3), activation='relu'))
model.add(tfl.MaxPooling3D((2, 2, 2)))
model.add(tfl.BatchNormalization())

model.add(tfl.Conv3D(128, (3, 3, 3), activation='relu'))
model.add(tfl.MaxPooling3D((2, 2, 2)))
model.add(tfl.BatchNormalization())

model.add(tfl.Conv3D(256, (3, 3, 3), activation='relu'))
model.add(tfl.MaxPooling3D((2, 2, 2)))
model.add(tfl.BatchNormalization())

model.add(tfl.GlobalAveragePooling3D())
model.add(tfl.Dense(512, activation='relu'))
model.add(tfl.Dropout(0.3))

model.add(tfl.Dense(1, activation='sigmoid'))

model.summary()

true_pos = tf.keras.metrics.TruePositives(name='tp')
true_neg = tf.keras.metrics.TrueNegatives(name='tn')
flse_pos = tf.keras.metrics.FalsePositives(name='fp')
flse_neg = tf.keras.metrics.FalseNegatives(name='fn')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'f1_score', true_pos, true_neg, flse_pos, flse_neg])
print(type(X_tr))
model.fit(x=X_tr, y=y_tr, epochs=2, batch_size=1, validation_data=(X_dv,y_dv), verbose=1)

#print(model.predict(X_dv))
model.save('cnn_1.keras')
