# Occlusion JEPA — contexte projet

Projet de recherche : permanence de l'objet dans un JEPA temporel. Disque
rebondissant 64×64 gris, barre verticale occlusive (position aléatoire par
épisode, occlusion totale ~3 frames). Voir README.md pour l'architecture.

## Question de recherche & état

La permanence de l'objet vit-elle dans les représentations (ẑ) ou dans le
prédicteur ? Le run GRU initial a montré qu'elle vivait dans l'état caché du
GRU (probe R² sur ẑ occlus : 0.26 vs 0.91 visible). Le prédicteur par défaut
est donc maintenant **markovien sans état** (`ẑ_t = ẑ_{t-1} + MLP(ẑ_{t-2}, ẑ_{t-1})`)
pour forcer la mémoire à passer par les latents. KPI : R² de la probe linéaire
sur les ẑ prédits pendant l'occlusion.

Leviers discutés non implémentés : masquage de la loss sur les pas occlus
(semi-supervisé, à n'activer que si le markovien ne suffit pas), GRU étranglé.

## Workflow

- **Entraînement sur Colab** (`train_colab.ipynb`, T4) ; l'utilisateur télécharge
  `best.pt` et le range dans `runs/<date_nom>/` avec figures + `probe_report.json`
  + `NOTES.md`. Un dossier par run, ne pas mélanger.
- **Éval/visu en local** : `viz_pca.ipynb` (pointer `CKPT_PATH` sur le run voulu),
  `scripts/data_viewer.py` pour inspecter les données.
- Python local : env conda `jepa-wms` (`~/miniconda3/envs/jepa-wms/bin/python`) —
  le python système n'a pas torch.
- Git : géré par l'utilisateur (commits/push). `*.pt` est gitignoré.

## Conventions & pièges

- Tout hyperparamètre vit dans `Config` (src/occlusion_jepa/config.py) ; `steps`
  est une **property** (= epochs × steps_per_epoch), ne pas en faire un champ.
- Données générées à la volée (IterableDataset infini) — pas de dataset stocké.
  Les sets d'éval sont figés par seed (`sample_batch`, val seed 987654).
- Calibration occlusion : durée ≈ `(bar_width − 2·disk_radius)/|vx|`. Vérifier
  avec `scripts/data_viewer.py --stats` après tout changement de ces paramètres.
- `load_checkpoint` doit rester rétrocompatible (il infère `predictor_type`
  depuis les poids pour les vieux ckpts GRU).
- Frames occluses d'un épisode = identiques entre elles → les z réels occlus ne
  peuvent pas contenir la position du disque (info-théoriquement impossible).
  Ne pas interpréter un R² faible sur z réels occlus comme un échec.
- Après modification du code : smoke test local (mini-config CPU : epochs=2,
  steps_per_epoch=15, batch_size=32, num_workers=0, val_seqs=32) couvrant
  train → checkpoint → probes → viz.
- Le training loss est 100% auto-supervisé : les flags `occluded`/`bar_x` des
  samples ne servent qu'à l'éval (et au forçage de traversée `crossing_frac`,
  qui ne change que la distribution des données).
