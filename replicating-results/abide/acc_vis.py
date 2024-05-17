import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.ndimage import gaussian_filter1d

# Path to csv of accuracies
csv_path = '/run/media/noah/TOSHIBA EXT/cnn_3-1_accs.csv'
sigma = 1

df = pd.read_csv(csv_path)
df = df.drop(columns=['Raw Train', 'Raw Dev', 'Raw OL'])
df['outlier'] = df['outlier'].astype(bool)
df = df.set_index('Epoch')

df['train_raw'] = df['train']
df['dev_raw'] = df['dev']
df['train'] = gaussian_filter1d(df['train_raw'], sigma=sigma)
df['dev'] = gaussian_filter1d(df['dev_raw'], sigma=sigma)

cm = sns.color_palette('tab20')
ncm = []
for i in [1, 3]:
    c = cm[i]
    nc = (c[0], c[1], c[2])
    ncm.append(nc)
    print(nc)

ndf = df[df['outlier'] == False].drop(columns=['outlier'])
epsilon = 0.01
nmin = ndf.min().min() - epsilon
nmax = ndf.max().max() + epsilon

style.use('seaborn-v0_8-talk')
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(ndf[['train_raw', 'dev_raw']], dashes=False, ax=ax, palette=ncm, legend=False, linewidth=1)
sns.lineplot(df[['train', 'dev']], dashes=False, ax=ax, palette='tab10')
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
ax.set_ylabel('Accuracy')
ax.set_title('Epoch Accuracies')
ax.set_ylim(nmin, nmax)
fig.savefig('accs.png')