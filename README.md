# Cluster2Former: Semisupervised Clustering Transformers for Video Instance Segmentation (Sensors 2024)

[Áron Fóthi](https://sciprofiles.com/profile/3300986), [Adrián Szlatincsán](https://sciprofiles.com/profile/3372387), [Ellák Somfai](https://sciprofiles.com/profile/3299837)

[[`mdpi`](https://www.mdpi.com/1424-8220/24/3/997)][[`BibTeX`](#CitingCluster2Former)]

<div align="center">
  <img src="https://drive.google.com/uc?id=1oQrZYdTT4PvycbOx_wuxTGcb2hYQsCVK" width="100%" height="100%"/>
  <img src="https://drive.google.com/uc?id=1tgrtDPd2leD0difHGY03Ula81jwZ4PCC" width="100%" height="100%"/>
</div>
<br/>

## Abstract

A novel approach for video instance segmentation is presented using semisupervised learning. Our Cluster2Former model leverages scribble-based annotations for training, significantly reducing the need for comprehensive pixel-level masks. We augment a video instance segmenter, for example, the Mask2Former architecture, with similarity-based constraint loss to handle partial annotations efficiently. We demonstrate that despite using lightweight annotations (using only 0.5% of the annotated pixels), Cluster2Former achieves competitive performance on standard benchmarks. The approach offers a cost-effective and computationally efficient solution for video instance segmentation, especially in scenarios with limited annotation resources.

Keywords: transformers; video processing; instance segmentation; semisupervised learning

### Features
* A single architecture for panoptic, instance and semantic segmentation.
* Based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), no change in the architecture of the model
* With the use of scribble like annotations and Similarity-based Constraint loss, you can achive competitive performance, but with much less annotation effort compared to the full mask annotation.
* [Tensorboard visualization](TENSORBOARD_VIS.md) support during training and evaluation
* Support major VIS datasets and [scribble version](https://github.com/szlAdrian/Scribble-Datasets-for-Segmentation) of them: YouTubeVIS 2019/2021, OVIS.

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for Mask2Former and Cluster2Former](datasets/README.md).

See [Getting Started with Mask2Former and Cluster2Former](GETTING_STARTED.md).

See more in [Mask2Former](https://github.com/facebookresearch/Mask2Former)

## Advanced usage

See [Advanced Usage of Mask2Former](ADVANCED_USAGE.md).

## Model Zoo and Baselines

We also provide a set of baseline results and trained models available for download in addition to the Model Zoo of the Mask2Fomer in the [Mask2Former and Cluster2Former Model Zoo](MODEL_ZOO.md).

## <a name="CitingCluster2Former"></a>Citing Cluster2Former

If you use Cluster2Former in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@Article{s24030997,
AUTHOR = {Fóthi, Áron and Szlatincsán, Adrián and Somfai, Ellák},
TITLE = {Cluster2Former: Semisupervised Clustering Transformers for Video Instance Segmentation},
JOURNAL = {Sensors},
YEAR = {2024},
}
```

## Acknowledgement

Code is based on Mask2Former (https://github.com/facebookresearch/Mask2Former).
