## Que faire ?

Bloquer la lecture des triggers digitaux, car absents.

## Comment ?

Travailler sur la matrice analog_in et digital_in. Dans le .json, chercher la clé "Version", si == "v2", modifier
DIGITAL_TRIGGERS_MAPPING en {"Basler": 0}

Se baser sur session_XXXX.json, meilleure souplesse.

Itérer sur les clés "Block_XXX"

Récupérer les fichiers de fréquences, les fichiers de positions.

# Gestion des blocs non complets

Itérer sur chaque bloc avec la fonction dans extraction_utils.py "check_if_block_complete"

Considérer par défaut que la suite de la session est morte, se concentrer sur les premiers blocs.

