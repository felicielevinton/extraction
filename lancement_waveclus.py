#wave_clus
#wave_clusdepuispython
#4pipeline_create_data
#pipeline_get_tono_spike

import subprocess
import time

# Liste des scripts à exécuter et des temps d'attente correspondants (en secondes)
scripts = [
    ('C:/Users/PC/Documents/spike-sorting/analysebingobike/wave_clus.py', 2),  # Attente de 5 secondes avant le prochain script
    ('C:/Users/PC/Documents/spike-sorting/analysebingobike/waveclusdepuispython.py', 3000),  # Attente de 10 secondes
    ('C:/Users/PC/Documents/spike-sorting/extraction/pipeline_create_npy.py', 2),  # Attente de 7 secondes
    ('C:/Users/PC/Documents/spike-sorting/extraction/pipeline_get_tono_spike_sorting.py', 0)  # Attente de 12 secondes
]

# Parcourir les scripts
for script, wait_time in scripts:
    try:
        print(f"Lancement de {script}...")
        # Exécuter le script Python
        subprocess.run(['python', script], check=True)
        print(f"{script} terminé.")
        
        # Attendre avant de passer au script suivant
        print(f"Attente de {wait_time} secondes avant le prochain script.")
        time.sleep(wait_time)
    
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {script}: {e}")
        break  # Arrête l'exécution si une erreur survient
