# Occlusion JEPA

Étude de la **permanence de l'objet** dans les représentations d'un JEPA temporel.

Un disque rebondit sur un écran 64×64 en niveaux de gris. Une barre verticale fixe
l'occulte totalement pendant ~3 frames lors de sa traversée. On entraîne un JEPA
(encodeur CNN + prédicteur GRU en rollout autorégressif, cible EMA, régularisation
VICReg) à prédire les embeddings des frames futures, puis on regarde :

1. **PCA 3D** — la représentation apprise est-elle interprétable à l'œil nu ?
2. **Interpolation latente** — le prédicteur "traverse-t-il" l'occlusion de façon
   lisse (le disque continue d'exister en latent) ?

## Structure

```
train_colab.ipynb          # notebook Colab : setup → train → éval
src/occlusion_jepa/
├── config.py              # tous les hyperparamètres
├── data.py                # génération à la volée (disque rebondissant + barre)
├── models.py              # Encoder CNN, Predictor GRU, Decoder
├── losses.py              # loss JEPA (invariance) + VICReg (variance/covariance)
├── train.py               # boucle d'entraînement JEPA + update EMA
├── probes.py              # décodeurs post-hoc + probe linéaire (x, y)
└── viz.py                 # PCA 3D, rollouts décodés, métriques d'occlusion
```

## Usage (Colab)

Ouvrir `train_colab.ipynb` dans Colab (GPU T4 suffisant) et exécuter les cellules :
clone du repo, sanity-check des données, entraînement (~30k steps), puis probes et
visualisations.

## Usage (local)

```bash
pip install -r requirements.txt
python -m occlusion_jepa.train --steps 200   # smoke test
```
