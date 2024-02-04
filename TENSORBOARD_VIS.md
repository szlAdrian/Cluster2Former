## Tensorboard visualization support

Next to the losses and other informations Cluster2Former supports visualization of the ground truth and the predictions of the model during training and evaluation.

#### Hyperparameters:
- `cfg.TB_VISUALIZATION.VIS_PERIOD` - Put visualization after that many iterations 
- `cfg.TB_VISUALIZATION.VIS_CONF_THRESHOLD` - Visualize only those predictions which has higher score the this

If you use the given apptainer container, then see the starting of the tensorboard in the [container_launcher.sh](container_launcher.sh) 

In the tensorboard on the top click to the 'IMAGE' button to see the visualizations e.g.:

<div align="center">
  <img src="https://drive.google.com/uc?id=1AcuZXMttp2t4mofCZs0MhlhleK0kdf2S" width="100%" height="100%"/>
  <img src="https://drive.google.com/uc?id=1RbdDEJn5VADtwMS4VCKtGcfUYAcwG3r0" width="100%" height="100%"/>
  <img src="https://drive.google.com/uc?id=1R3S13DcunFSmWer64nh5eWnIjwDnh0eC" width="100%" height="100%"/>
</div>