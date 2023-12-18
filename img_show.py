# =======================================================
# MODULE
# =======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =======================================================
# FUNCTION
# =======================================================

def my_function(value):
    return tuple(map(int, value.split(', ')))

# =======================================================
# LOAD DATA
# =======================================================

df = pd.read_csv('data/pano_colors.csv')
df.columns

# =======================================================
# PARAMETER
# =======================================================

# Output dir
dir_fig = 'figs'
# Créer le dossier s'il n'existe pas
os.makedirs(dir_fig, exist_ok=True)

# Le paramètre que tu veux regarder
parameter = 'kmeans_1'

# =======================================================
# PRE-PROCESS
# =======================================================

exclude_column = 'img'

# Conversion des couleurs en tuples
string_columns = df.select_dtypes(include='object').columns.tolist()
string_columns = [col for col in string_columns if col != exclude_column]
df[string_columns] = df[string_columns].apply(lambda x: x.str.split(', ').apply(lambda y: tuple(map(int, y))))

# Création de dataframes filter par images pour la figure
pattern = r'.*_1_.*'  
df_1 = df[df['img'].str.contains(pattern)]
df_1.reset_index(inplace=True, drop=True)
pattern = r'.*_2_.*'  
df_2 = df[df['img'].str.contains(pattern)]
df_2.reset_index(inplace=True, drop=True)
pattern = r'.*_3_.*'  
df_3 = df[df['img'].str.contains(pattern)]
df_3.reset_index(inplace=True, drop=True)

# =======================================================
# PLOT
# =======================================================

fig, axs = plt.subplots(3, len(df_1.kmeans_1), figsize=(10, 5)) # !! ICI !!

for i, ax in enumerate(axs[0]):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=np.array(df_1[f'{parameter}'][i]) / 255.0))
    ax.axis('off')

for j in range(1, 3):
    for i, ax in enumerate(axs[j]):
        if j == 1:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=np.array(df_2[f'{parameter}'][i]) / 255.0))
        elif j == 2:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=np.array(df_3[f'{parameter}'][i]) / 255.0))
        ax.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)

# Sauvegarder la figure et la montrer
plt.savefig(f'{dir_fig}/frise_{parameter}.png', dpi=300)
plt.show()
