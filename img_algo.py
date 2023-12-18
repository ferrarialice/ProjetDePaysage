# =======================================================
# MODULE
# =======================================================
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import pandas as pd

# =======================================================
# PARAMETER
# =======================================================

# Dossier contenant les images
img_dir = "img"

# =======================================================
# PROCESS
# =======================================================

dict_color = {}
for file in os.listdir(img_dir):
    print(file)
    base_name, extension = os.path.splitext(file)

    # Créer le chemin vers le fichier
    img_path = f"{img_dir}/{file}"
    img = cv2.imread(img_path)

    # Couper l'image en 9 parties
    height, width, _ = img.shape

    tier_height = height // 3
    tier_width = width // 3

    # Dictionnaire pour enregistrer les images découpées
    output_dir = f'subset'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the tiers and extract the corresponding image region
    for i in range(3):
        # Calculate the starting and ending vertical coordinates of the tier
        start_y = i * tier_height
        end_y = (i + 1) * tier_height

        # Extract the image region corresponding to the current tier
        tier_image = img[start_y:end_y, :]

        # Iterate over the sub-regions horizontally within the tier
        for j in range(3):
            # Définir le nom de l'image découpée courante
            name_sub = f'{base_name}_{i + 1}_{j + 1}.jpg'
            name_sub_wext = f'{base_name}_{i + 1}_{j + 1}'
            print(name_sub)

            # Calculate the starting and ending horizontal coordinates of the sub-region
            start_x = j * tier_width
            end_x = (j + 1) * tier_width

            # Extract the sub-region within the current tier
            subregion_image = tier_image[:, start_x:end_x]
            print("Cut : done")

            # --------------------------------------------------------
            # Couleur moyenne
            # --------------------------------------------------------
            average_color = np.mean(subregion_image, axis=(0, 1))
            average_color = np.round(average_color[::-1]).astype(int)
            average_color = [str(value) for value in average_color]
            average_color = ', '.join(average_color)
            print("Average : done")

            # --------------------------------------------------------
            # Most frequent pixel
            # --------------------------------------------------------
            pixels = subregion_image.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

            sorted_indices = np.argsort(counts)[::-1]
            sorted_colors = unique_colors[sorted_indices]
            sorted_counts = counts[sorted_indices]

            top_colors = []
            top_percentages = []

            for k in range(min(3, len(sorted_colors))):
                color = sorted_colors[k]
                color = np.round(color[::-1]).astype(int)
                count = sorted_counts[k]
                percentage = (count / len(pixels)) * 100
                color_str = ', '.join(str(value) for value in color)
                top_colors.append(color_str)
                top_percentages.append(percentage)

            # Enregistrer dans des variables les valeurs d'intérêt
            freq_1 = top_colors[0]
            freq_1_pct = top_percentages[0]
            freq_2 = top_colors[1]
            freq_2_pct = top_percentages[1]
            freq_3 = top_colors[2]
            freq_3_pct = top_percentages[2]
            print("Dominant - frequency : done")

            # --------------------------------------------------------
            # K-MEANS
            # --------------------------------------------------------
            # Dossier contenant les logs du processus des k-means
            dir_log = "log_kmeans"
            os.makedirs(dir_log, exist_ok=True)
            log_file = open(f"{dir_log}/{name_sub_wext}_kmeans_log.txt", "w")

            # Define the number of clusters
            num_clusters = 3

            # Perform k-means clustering
            clt = KMeans(n_clusters=num_clusters)
            clt.fit(pixels.reshape(-1, 3))
            _, counts = np.unique(clt.labels_, return_counts=True)
            cluster_count = np.max(counts)

            # Get the dominant colors
            dominant_colors = clt.cluster_centers_

            # Sort the dominant colors based on cluster counts
            sorted_indices = np.argsort(counts)[::-1]
            dominant_colors = dominant_colors[sorted_indices]
            sorted_counts = counts[sorted_indices]

            # Retrieve the top dominant colors
            k_top_col = []
            k_top_pct = []

            for l in range(min(3, len(dominant_colors))):
                k_col = dominant_colors[l]
                k_col = np.round(k_col[::-1]).astype(int)  # Voilà !!!
                k_col = [str(value) for value in k_col]
                k_col_str = ', '.join(k_col)

                count = sorted_counts[l]
                k_pct = (count / len(pixels)) * 100

                k_top_col.append(k_col_str)
                k_top_pct.append(k_pct)

            # Write to log file
            log_file.write(f"Number of Clusters: {num_clusters}\n")
            log_file.write(f"Cluster Centers:\n{clt.cluster_centers_}\n")
            log_file.write(f"Cluster Sizes:\n{counts}\n")
            log_file.write(f"WCSS: {clt.inertia_}\n\n")

            # Store the dominant colors as string variables
            kmeans_1 = k_top_col[0]
            kmeans_1_pct = k_top_pct[0]
            kmeans_2 = k_top_col[1]
            kmeans_2_pct = k_top_pct[1]
            kmeans_3 = k_top_col[2]
            kmeans_3_pct = k_top_pct[2]
            # kmeans_4 = k_top_col[3]
            # kmeans_4_pct = k_top_pct[3]
            # kmeans_5 = k_top_col[4]
            # kmeans_5_pct = k_top_pct[4]

            # Write to log file
            log_file.write("Dominant Colors:\n")
            log_file.write(f"1st Dominant Color: {kmeans_1} ({kmeans_1_pct} %)\n")
            log_file.write(f"2nd Dominant Color: {kmeans_2} ({kmeans_2_pct} %)\n")
            log_file.write(f"3rd Dominant Color: {kmeans_3} ({kmeans_3_pct} %)\n")
            # log_file.write(f"4th Dominant Color: {kmeans_4} ({kmeans_4_pct} %)\n")
            # log_file.write(f"5th Dominant Color: {kmeans_5} ({kmeans_5_pct} %)\n")

            # Close the log file
            log_file.close()
            print("Dominant - k-means : done")

            # Aggréger les données calculées dans un df
            df_color_sub = pd.DataFrame({
                'img': name_sub,
                'mean': average_color,
                'freq_1': freq_1,
                'freq_1_pct': freq_1_pct,
                'freq_2': freq_2,
                'freq_2_pct': freq_2_pct,
                'freq_3': freq_3,
                'freq_3_pct': freq_3_pct,
                'kmeans_1': kmeans_1,
                'kmeans_1_pct': kmeans_1_pct,
                'kmeans_2': kmeans_2,
                'kmeans_2_pct': kmeans_2_pct,
                'kmeans_3': kmeans_3,
                'kmeans_3_pct': kmeans_3_pct
            }, index=[0])

            # Passer le df dans le dictionnaire
            dict_color[name_sub] = df_color_sub

            # Enregistrer la photo (1/9)
            output_path = os.path.join(output_dir, f'{base_name}_{i + 1}_{j + 1}.jpg')
            cv2.imwrite(output_path, subregion_image)

# Joindre toutes données
df_all = pd.concat(dict_color, ignore_index=True)

# Créer le fichier
data_dir = f'data'
os.makedirs(data_dir, exist_ok=True)
df_all.to_csv(f'{data_dir}/pano_colors.csv', index=False)
