import os

import tensorflow as tf
import tensorflow.keras.layers as tfl
import nibabel as nib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd

NIIROOT = '/media/noah/TOSHIBA EXT/Outputs/cpac/nofilt_noglobal/reho'
CSVPATH = 'replicating-results/abide/subject_data.csv'

niipaths = []
for root, dirs, files in os.walk(NIIROOT):
    for file in files:
        if file.endswith('.gz'):
            niipaths.append((root, file))

imgs = [(p[1][:-12], (nib.load(os.path.join(p[0], p[1])))) for p in tqdm(niipaths, desc='load imgs')]
df = pd.read_csv(CSVPATH)

y = np.array([[list(df[df['FILE_ID'] == x[0]]['DX_GROUP'])[0]-1 for x in imgs]]).astype(np.float32).T

X = [np.array(img[1].get_fdata()).astype(np.float32) for img in tqdm(imgs, desc='load data')]

model = tf.keras.Sequential(name='abide_cnn')

X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.4, random_state=202410350)
X_dv, X_te, y_dv, y_te = train_test_split(X_vl, y_vl, random_state=350202410)

X_tr = tf.constant(np.array(X_tr))
y_tr = tf.constant(np.array(y_tr))
X_dv = tf.constant(np.array(X_dv))
y_dv = tf.constant(np.array(y_dv))
X_te = tf.constant(np.array(X_te))
y_te = tf.constant(np.array(y_te))

print(f"--------\nDATA SHAPES:\n tr:\n  x: {X_tr.shape} {X_tr.dtype}\n  y: {y_tr.shape} {y_tr.dtype}\n dv:\n  x: {X_dv.shape} {X_dv.dtype}\n  y: {y_dv.shape} {y_dv.dtype}\n te:\n  x: {X_te.shape} {X_te.dtype}\n  y: {y_te.shape} {y_te.dtype}\n--------")

# Model architecture copied from Alharthi & Alzahrani (11/2023)
# DOI: 10.3390/brainsci13111578

model.add(tfl.InputLayer(shape=(61, 73, 61, 1)))

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

class MeanPrediction(tf.keras.metrics.Metric):
    def __init__(self, name="mp", **kwargs):
        super(MeanPrediction, self).__init__(name=name, **kwargs)
        self.mean_prediction = self.add_weight(name="mp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mean_prediction.assign(tf.reduce_mean(y_pred))

    def result(self):
        return self.mean_prediction

bin_acc = tf.keras.metrics.BinaryAccuracy(name='acc')
true_pos = tf.keras.metrics.TruePositives(name='tp')
true_neg = tf.keras.metrics.TrueNegatives(name='tn')
flse_pos = tf.keras.metrics.FalsePositives(name='fp')
flse_neg = tf.keras.metrics.FalseNegatives(name='fn')

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[bin_acc, true_pos, true_neg, flse_pos, flse_neg, MeanPrediction()])

checker = tf.keras.callbacks.ModelCheckpoint('/media/noah/TOSHIBA EXT/cnn_cp.keras', monitor='val_loss', verbose=1)
model.fit(x=X_tr, y=y_tr, epochs=50, batch_size=1, validation_data=(X_dv,y_dv), verbose=1, callbacks=[checker])

model.save('/media/noah/TOSHIBA EXT/cnn.keras')
