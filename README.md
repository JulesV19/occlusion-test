# Occlusion JEPA

Étude de la **permanence de l'objet** dans les représentations d'un JEPA temporel.

Un disque rebondit sur un écran 64×64 en niveaux de gris. Une barre verticale —
position tirée aléatoirement **à chaque épisode** (fixe au sein d'un épisode) —
l'occulte totalement pendant ~3 frames lors de sa traversée. On entraîne un JEPA
à prédire les embeddings des frames futures en rollout latent, puis on regarde :

1. **PCA 3D** — la représentation apprise est-elle interprétable à l'œil nu ?
2. **Interpolation latente** — le prédicteur "traverse-t-il" l'occlusion de façon
   lisse (le disque continue d'exister en latent) ?

## Architecture

- **Encodeur** : petit CNN (GroupNorm) → embedding 128 dims, par frame.
- **Cible** : encodeur EMA (momentum 0.996 → 1.0) + stop-gradient, style I-JEPA.
- **Prédicteur** (défaut) : **markovien sans état** — `ẑ_t = ẑ_{t-1} + MLP(ẑ_{t-2}, ẑ_{t-1})`.
  Aucun état caché privé : la seule mémoire qui traverse une occlusion est le
  latent lui-même. Baseline récurrente : `Config(predictor_type="gru")` (un GRU
  peut porter la permanence dans son état caché au lieu des ẑ — c'est exactement
  ce qu'on a observé, d'où le markovien par défaut).
- **Anti-collapse** : VICReg (variance + covariance) sur les embeddings online.
- **Entraînement** : rollout multi-step (4 frames de contexte → 8 pas prédits),
  données générées à la volée (dataset infini), epochs de 500 steps avec set de
  validation figé, `best.pt`/`last.pt` sauvegardés en continu.

## Structure

```
train_colab.ipynb          # notebook Colab : setup → train → probes → rollouts décodés
viz_pca.ipynb              # visualisations PCA 3D (local ou Colab, charge un best.pt)
scripts/data_viewer.py     # viewer local des données (interactif, --stats, --gif)
runs/<date_nom>/           # un dossier par run : best.pt, probe_report.json, NOTES.md, figures/
src/occlusion_jepa/
├── config.py              # tous les hyperparamètres (Config dataclass)
├── data.py                # génération à la volée (disque + barre mobile par épisode)
├── models.py              # Encoder, MarkovPredictor / GRUPredictor, Decoder
├── losses.py              # loss JEPA (invariance) + VICReg (variance/covariance)
├── train.py               # boucle par epochs, EMA, val figée, best.pt/last.pt
├── probes.py              # décodeurs post-hoc (obs + oracle) + probe linéaire (x, y)
└── viz.py                 # PCA 3D, métriques latentes, rollouts décodés
```

## Usage

**Colab** (entraînement) : ouvrir `train_colab.ipynb`, runtime GPU, exécuter.
Les checkpoints `best.pt`/`last.pt` sont téléchargeables en cours de run.

**Local** (éval/visu, env avec torch) :
```bash
pip install -r requirements.txt
python scripts/data_viewer.py            # inspecter les données (--stats pour la calibration)
# ouvrir viz_pca.ipynb et pointer CKPT_PATH vers runs/<run>/best.pt
python -m occlusion_jepa.train --epochs 2 --steps-per-epoch 50   # smoke test
```

## Métriques de succès

- Probe linéaire z → (x, y) : R² **sur les ẑ prédits pendant l'occlusion**
  (les z réels occlus ne peuvent pas contenir la position : frames identiques).
- Rollouts décodés (decoder-oracle) : le disque décodé depuis ẑ est-il à la bonne
  position pendant et après la traversée ?
- PCA 3D : nappe continue suivant (x, y) ; trajectoire prédite qui ne s'écrase
  pas sur la zone "frames vides" pendant l'occlusion.
