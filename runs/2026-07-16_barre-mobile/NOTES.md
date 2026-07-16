# Run 2026-07-16 — barre mobile

Premier run avec barre à position aléatoire par épisode. 60 epochs × 500 steps,
Colab T4. Config : bar_width=20, bar_margin=12, speed_x=(2.5, 4.5) → occlusion ~3 frames.

## Fichiers
- `best.pt` — meilleur checkpoint (val_inv), rechargeable via `load_checkpoint`
- `probe_report.json` — probe linéaire z → (x, y)
- `figures/01_history.png` — courbes par epoch (inv train/val/occlus, VICReg, cosine, z_std)
- `figures/02_rollouts_decodes.png` — rollouts décodés (obs + oracle)
- `figures/03_metriques_latentes.png` — MSE/cosine par pas de rollout

## Lecture rapide
- Pas de collapse (z_std ≈ 1.05 stable), val_cos ~0.87 en fin de run.
- Probe sur z réels : visible MAE 1.74 px (R² 0.97) / occlus MAE 7.7 px (R² 0.14)
  → attendu : les frames occluses d'un épisode sont identiques entre elles, la
  position ne PEUT PAS être dans l'embedding (le R² > 0 vient de bar_x, qui borne x).
- Probe sur ẑ prédits : visible MAE 2.99 px (R² 0.91) / occlus MAE 7.1 px (R² 0.26)
  → le rollout transporte bien la position pour les frames visibles, et les ẑ
  occlus en retiennent un peu plus que les z réels occlus.
- Rollouts décodés : pendant l'occlusion l'oracle ne décode pas de disque
  (fantôme de barre), mais le disque RÉAPPARAÎT au bon endroit après
  → la permanence de l'objet vit dans la mémoire du GRU, pas dans l'embedding.
