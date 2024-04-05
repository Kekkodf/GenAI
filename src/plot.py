import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

realResponses = pd.read_csv('data/obfuscated/obfuscatedTestRun_CMP_5.csv')
# Keep only unique ids
realResponses = realResponses.drop_duplicates(subset='question').reset_index(drop=True)

eps = [5, 10, 20, 30, 40, 50]
models = ['llama2-70b-4096', 'mixtral-8x7b-32768', 'gemma-7b-it']

# Initialize a dictionary to store average precision values for each model and mechanism
avg_precision_cmp = {model: [] for model in models}
avg_precision_mahalanobis = {model: [] for model in models}

df = pd.DataFrame(columns=["model", "mechanism", "epsilon", "precision"])
for model in models:
    for e in eps:
        mechanism = 'CMP'
        path_cmp = f'output/{model}/{mechanism}/{mechanism}_{e}.csv'
        df_cmp = pd.read_csv(path_cmp)
        responses_cmp = df_cmp['response']
        # Compare the responses with the real responses for CMP mechanism
        realResponses['response_cmp'] = responses_cmp
        realResponses['correct_cmp'] = realResponses.apply(lambda x: x['response_cmp'] == x['answer'], axis=1)
        # Compute precision (accuracy) for CMP mechanism
        precision_cmp = realResponses['correct_cmp'].mean()
        avg_precision_cmp[model].append(precision_cmp)

        temp = pd.DataFrame({
            "model": [model],
            "mechanism": [mechanism],
            "epsilon": [e],
            "precision": [precision_cmp]
        })
        # Append the results to the dataframe
        df = pd.concat([df, temp], ignore_index=True)

        mechanism = 'Mahalanobis'
        path_mahalanobis = f'output/{model}/{mechanism}/{mechanism}_{e}.csv'
        df_mahalanobis = pd.read_csv(path_mahalanobis)
        responses_mahalanobis = df_mahalanobis['response']
        # Compare the responses with the real responses for Mahalanobis mechanism
        realResponses['response_mahalanobis'] = responses_mahalanobis
        realResponses['correct_mahalanobis'] = realResponses.apply(lambda x: x['response_mahalanobis'] == x['answer'], axis=1)
        # Compute precision (accuracy) for Mahalanobis mechanism
        precision_mahalanobis = realResponses['correct_mahalanobis'].mean()
        avg_precision_mahalanobis[model].append(precision_mahalanobis)

        temp = pd.DataFrame({
            "model": [model],
            "mechanism": [mechanism],
            "epsilon": [e],
            "precision": [precision_mahalanobis]
        })
        # Append the results to the dataframe
        df = pd.concat([df, temp], ignore_index=True)

# Save the results to a CSV file
df.to_csv('output/average_precision.csv', index=False)

plt.figure(figsize=(15, 8))
colors = sns.color_palette('colorblind', n_colors=len(models))

for i, model in enumerate(models):
    color = colors[i]
    if model == 'llama2-70b-4096':
        model_label = 'Llama2'
    elif model == 'gemma-7b-it':
        model_label = 'Gemma'
    elif model == 'mixtral-8x7b-32768':
        model_label = 'Mixtral'
    else:
        model_label = model
    
    plt.plot(eps, avg_precision_cmp[model], linestyle=':', marker='o', color=color, label=f'{model_label} (CMP)')
    plt.plot(eps, avg_precision_mahalanobis[model], linestyle='-', marker='s', color=color, label=f'{model_label} (Mahalanobis)')

plt.xlabel(r'$\varepsilon$')
plt.ylabel('Average Precision')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.82), fancybox=False, shadow=False, ncol=2)
plt.tight_layout()
plt.grid(False)

# Save the plot as a PDF file
plt.savefig('output/average_precision_plot.pdf', bbox_inches='tight', format='pdf')
plt.show()
