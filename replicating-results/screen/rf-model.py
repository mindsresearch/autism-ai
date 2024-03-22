import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# argparse stuff and setup...
#
ARGSTR = '-c data/'

parser = argparse.ArgumentParser(prog='first-model')
parser.add_argument('-c', help='path/to/csvs/')
args = parser.parse_args(ARGSTR.split())

def enum_files(rootpath:str, ext:str='csv', blacklist:list=None) -> list:
    fl = []
    for root, dirs, files in os.walk(rootpath): # pylint: disable=unused-variable
        for file in files:
            if file.endswith(f'.{ext}'):
                if (blacklist is not None) and (any(bad in root for bad in blacklist)):
                    print(f'denied by BL:\n {root}')
                else:
                    fl.append(os.path.join(root, file))
    return fl

# data loading and splitting...
#
df = [pd.read_csv(path, true_values=['Yes', 'YES', 'yes'], false_values=['no', 'No', 'NO'], na_values=['?']) for path in enum_files(args.c)]
df = pd.concat(df, ignore_index=True)
df.drop(['Unnamed: 0', 'Age_Mons', 'Case_No', 'age_desc'], axis=1, inplace=True)
df.replace({False: 0, True: 1}, inplace=True)
y = df['class'].to_numpy()
x = df.drop(['class', 'gender', 'ethnicity', 'contry_of_res', 'relation'], axis=1)
ohe = OneHotEncoder()
temp = ohe.fit_transform(df[['gender', 'ethnicity', 'contry_of_res', 'relation']]).toarray()
temp = pd.DataFrame.from_records(temp, columns=ohe.get_feature_names_out())
x = pd.concat([x, temp], axis=1)
x_tr, x_vl, y_tr, y_vl = tts(x, y, test_size=0.4)
x_dv, x_te, y_dv, y_te = tts(x_vl, y_vl)
print(f'--------\nDATA SHAPES:\n train: {x_tr.shape}\n   dev: {x_dv.shape}\n  test: {x_te.shape}\n--------')

# model creation
#
# Model pipeline copied from: Shrivastava et al. (Mar. 2024)
# DOI: 10.1007/s13755-024-00277-8
tfrs = [('imputer', KNNImputer()),
        ('scaler', MinMaxScaler()),
        ('forest', RandomForestClassifier())
        ]

pipe = Pipeline(tfrs, verbose=True)

# model run
#
pipe = pipe.fit(x_tr, y_tr)
p_tr = pipe.predict(x_tr)
p_dv = pipe.predict(x_dv)

# model eval
#
acc_tr = accuracy_score(y_tr, p_tr)
cmt_tr = confusion_matrix(y_tr, p_tr)
acc_dv = accuracy_score(y_dv, p_dv)
cmt_dv = confusion_matrix(y_dv, p_dv)

print(f'--------\nMODEL EVAL:\n TRAIN:\n  acc: {acc_tr}\n  conf. matrix:\n{cmt_tr}\n\n DEV:\n  acc: {acc_dv}\n  conf. matrix:\n{cmt_dv}\n--------')