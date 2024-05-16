import os

import tensorflow as tf
import tensorflow.keras.layers as tfl
import nibabel as nib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

NIIROOT = '/run/media/noah/TOSHIBA EXT/Outputs/cpac/nofilt_noglobal/reho'
CSVPATH = 'subject_data.csv'
MODELNO = '3-1'

niipaths = []
for root, dirs, files in os.walk(NIIROOT):
    for file in files:
        if file.endswith('.gz'):
            niipaths.append((root, file))

imgs = [(p[1][:-12], (nib.load(os.path.join(p[0], p[1])))) for p in tqdm(niipaths, desc='load imgs')]
df = pd.read_csv(CSVPATH)

y = np.array([[list(df[df['FILE_ID'] == x[0]]['DX_GROUP'])[0]-1 for x in imgs]]).astype(np.float32).T

X = [np.array(img[1].get_fdata()).astype(np.float32) for img in tqdm(imgs, desc='load data')]

model = tf.keras.Sequential(name=f'abide_cnn_{MODELNO}')

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

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

logdir = f'/run/media/noah/TOSHIBA EXT/cnn_{MODELNO}_logs'
# Define the basic TensorBoard callback.
boarder = tf.keras.callbacks.TensorBoard(log_dir=logdir)
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

def build_violin(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    train_pred = np.array(model.predict(X_tr)).flatten()
    dev_pred = np.array(model.predict(X_dv)).flatten()
    g = np.repeat(['train','dev'],[len(y_tr), len(y_dv)])
    df = pd.DataFrame({
        'pred': np.concatenate([train_pred, dev_pred]),
        'true': np.concatenate([np.array(y_tr).flatten(), np.array(y_dv).flatten()]),
        'group': g
        })

    # Log the confusion matrix as an image summary.
    vp = sns.violinplot(data=df, x='true', y='pred', hue='group', split=True, orient='v', inner=None)
    vp.set_title(f'Epoch {epoch}')
    fig = vp.get_figure()
    v_image = plot_to_image(fig)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("epoch preds", v_image, step=epoch)

# Define the per-epoch callback.
violiner = tf.keras.callbacks.LambdaCallback(on_epoch_end=build_violin)

bin_acc = tf.keras.metrics.BinaryAccuracy(name='acc')
true_pos = tf.keras.metrics.TruePositives(name='tp')
true_neg = tf.keras.metrics.TrueNegatives(name='tn')
flse_pos = tf.keras.metrics.FalsePositives(name='fp')
flse_neg = tf.keras.metrics.FalseNegatives(name='fn')

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[bin_acc, true_pos, true_neg, flse_pos, flse_neg, MeanPrediction()])

checker = tf.keras.callbacks.ModelCheckpoint(f'/run/media/noah/TOSHIBA EXT/cnn_{MODELNO}_cp.keras', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, save_freq='epoch')
model.fit(x=X_tr, y=y_tr, epochs=50, batch_size=1, validation_data=(X_dv,y_dv), verbose=1, callbacks=[checker, boarder, violiner])

model.save(f'/run/media/noah/TOSHIBA EXT/cnn_{MODELNO}.keras')
