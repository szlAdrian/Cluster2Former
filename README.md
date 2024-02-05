# Cluster2Former: Semisupervised Clustering Transformers for Video Instance Segmentation (Sensors 2024)

[Áron Fóthi](https://sciprofiles.com/profile/3300986), [Adrián Szlatincsán](https://sciprofiles.com/profile/3372387), [Ellák Somfai](https://sciprofiles.com/profile/3299837)

[[`mdpi`](https://www.mdpi.com/1424-8220/24/3/997)]

<div align="center">
  <img src="https://drive.google.com/uc?id=1oQrZYdTT4PvycbOx_wuxTGcb2hYQsCVK" width="100%" height="100%"/>
</div>

<div style="display: flex;">
  <img src="https://drive.google.com/uc?id=1oqzV1tfF-DxWzFNUMnSn9aVXqIbcrCUl" style="width: 50%; max-width: 50%; height: auto;"/>
  <img src="https://drive.google.com/uc?id=1B1xC5sEs7C28GmhbvH7AUAwS3-uxtIJw" style="width: 50%; max-width: 50%; height: auto;"/>
</div>
<br/>

### Features
* A single architecture for panoptic, instance and semantic segmentation.
* Based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), no change in the architecture of the model
* With the use of scribble like annotations and Similarity-based Constraint loss, you can achive competitive performance, but with much less annotation effort compared to the full mask annotation.
* [Tensorboard visualization](TENSORBOARD_VIS.md) support during training and evaluation
* Support major VIS datasets and [scribble version](https://github.com/szlAdrian/Scribble-Datasets-for-Segmentation) of them: YouTubeVIS 2019/2021, OVIS.

## Installation

See [installation instructions](INSTALL.md).

## Acknowledgement

Code is based on Mask2Former (https://github.com/facebookresearch/Mask2Former).