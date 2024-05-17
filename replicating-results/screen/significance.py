from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import style
import numpy as np
import seaborn as sns
from tqdm import tqdm

import rf_model

pipe, dataset = rf_model.create_model('data')

do_test = input('Run test? (y/n): ')

if do_test.lower() == 'y':
    x_te, y_te = dataset['test']
    x_tr, y_tr = dataset['train']
    p_te = pipe.predict(x_te)
    acc_te = accuracy_score(y_te, p_te)
    cmt_te = confusion_matrix(y_te, p_te)
    print(f'--------\nTEST EVAL:\n acc: {acc_te}\n conf. matrix:\n{cmt_te}\n--------')
    temp = x_te.copy()
    temp['actual'] = y_te
    temp['predicted'] = p_te
    temp.to_csv('test_results.csv', index=True)

do_imp = input('Run feature importance? (y/n): ')

if do_imp.lower() == 'y':
    # Extract feature importances from the 'forest' step of the pipeline
    feature_importances = pipe.named_steps['forest'].feature_importances_
    features = x_tr.columns

    # Sort them in descending order and prepare for plotting
    indices = np.argsort(feature_importances)[::-1]
    sorted_features = [features[i] for i in indices]
    sorted_importances = feature_importances[indices]

    # Assign colors based on specific feature groups
    colors = ['A-Score' if 'Score' in f else 'Ethnic group' if 'ethnicity' in f else 'Examiner' if 'relation' in f else 'Gender' if 'gender' in f else 'Other' for f in sorted_features]
    cmap = {'A-Score': '#86C338', 'Ethnic group': '#1590A8', 'Examiner': '#DA015C', 'Gender': '#0576C4', 'Other': '#FECE01'}
    # Plotting with new color distinctions
    fig, ax = plt.subplots(figsize=(12, 8))
    style.use('seaborn-v0_8-talk')
    ax.grid(True, axis='x', linestyle='--', lw=0.5)
    sns.barplot(y=sorted_features, hue=colors, ax=ax, palette=cmap, x=sorted_importances, orient='h')
    plt.title('Feature Importance by Group')
    fig.tight_layout()
    fig.savefig('feature_importance.png')

do_rep = input('Run repeated importances? (y/n): ')

if do_rep.lower() == 'y':
    import pandas as pd
    imps = {}
    n_iters = 50
    for i in tqdm(range(n_iters)):
        pipe, dataset = rf_model.create_model('data', verbose=False)
        x_tr, y_tr = dataset['train']
        features = x_tr.columns
        feature_importances = pipe.named_steps['forest'].feature_importances_
        for f, i in zip(features, feature_importances):
            if f in imps:
                imps[f].append(i)
            else:
                imps[f] = [i]
    df = pd.DataFrame(imps)
    df['run'] = [x+1 for x in range(n_iters)]
    df = pd.melt(df, id_vars='run', var_name='feature', value_name='importance')

    df['mean_importance'] = df.groupby('feature')['importance'].transform('mean')
    df = df.sort_values('mean_importance', ascending=False)

    df['group'] = ['A-Score' if 'Score' in f else 'Ethnic group' if 'ethnicity' in f else 'Examiner' if 'relation' in f else 'Gender' if 'gender' in f else 'Other' for f in df['feature']]
    df.to_csv('repeated_importances.csv', index=True)

    cmap = {'A-Score': '#86C338', 'Ethnic group': '#1590A8', 'Examiner': '#DA015C', 'Gender': '#0576C4', 'Other': '#FECE01'}
    fig, ax = plt.subplots(figsize=(12, 8))
    style.use('seaborn-v0_8-talk')
    ax.grid(True, axis='x', linestyle='--', lw=0.5)
    sns.barplot(data=df, x='importance', y='feature', hue='group', ax=ax, palette=cmap, orient='h', errorbar=('ci', 95))
    ax.set_title('Feature Importance by Group (95% CI)')
    fig.tight_layout()
    fig.savefig('repeated_importances.png')