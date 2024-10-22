import os
import re

# Spécifiez le chemin du répertoire où se trouvent vos fichiers
directory = 'Y:/eTheremin/clara/ALTAI_20240722_SESSION_02'

# Expression régulière pour extraire 'C<number>.mat'
pattern = re.compile(r'C\d+\.mat')

# Parcourir tous les fichiers du répertoire
for filename in os.listdir(directory):
    # Rechercher le motif 'C<number>.mat' dans le nom du fichier
    match = pattern.search(filename)
    
    if match:
        # Si un match est trouvé, on extrait le nom final 'C<number>.mat'
        new_name = match.group(0)
        
        # Construire les chemins complets vers les fichiers
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)

        # Renommer le fichier
        os.rename(old_file, new_file)
        print(f"Renommé : {old_file} -> {new_file}")
