<div  align="center">    
 <img src="resources/CLS.png" width = "400" height = "130" alt="segmentation" align=center />
</div>

## Introduction

RSI-Classification is an open source image classification toolbox based on PyTorch. 

The master branch works with **PyTorch 1.10+**.

Thanks to openMMlab, we have the following
### Major features

- Various backbones and pretrained models
- Bag of training tricks
- Large-scale training configs
- High efficiency and extensibility
- Powerful toolkits

## Installation

Below are quick steps for installation:

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
git clone https://github.com/EarthNets/RSI-Classification.git
cd RSI-Classification
pip3 install -e .
```


## Getting Started

Coming soon

## Model zoo

Results and models are available in the [model zoo].

<details open>
<summary>Supported backbones</summary>

- [x] [VGG](https://github.com/open-mmlab/mmclassification/tree/master/configs/vgg)
- [x] [ResNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)
- [x] [ResNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)
- [x] [SE-ResNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)
- [x] [SE-ResNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)
- [x] [RegNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/regnet)
- [x] [ShuffleNetV1](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v1)
- [x] [ShuffleNetV2](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v2)
- [x] [MobileNetV2](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v2)
- [x] [MobileNetV3](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v3)
- [x] [Swin-Transformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/swin_transformer)
- [x] [RepVGG](https://github.com/open-mmlab/mmclassification/tree/master/configs/repvgg)
- [x] [Vision-Transformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/vision_transformer)
- [x] [Transformer-in-Transformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/tnt)
- [x] [Res2Net](https://github.com/open-mmlab/mmclassification/tree/master/configs/res2net)
- [x] [MLP-Mixer](https://github.com/open-mmlab/mmclassification/tree/master/configs/mlp_mixer)
- [x] [DeiT](https://github.com/open-mmlab/mmclassification/tree/master/configs/deit)
- [x] [Conformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/conformer)
- [x] [T2T-ViT](https://github.com/open-mmlab/mmclassification/tree/master/configs/t2t_vit)
- [x] [Twins](https://github.com/open-mmlab/mmclassification/tree/master/configs/twins)
- [x] [EfficientNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/efficientnet)
- [x] [ConvNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/convnext)
- [x] [HRNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/hrnet)
- [x] [VAN](https://github.com/open-mmlab/mmclassification/tree/master/configs/van)
- [x] [ConvMixer](https://github.com/open-mmlab/mmclassification/tree/master/configs/convmixer)
- [x] [CSPNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/cspnet)
- [x] [PoolFormer](https://github.com/open-mmlab/mmclassification/tree/master/configs/poolformer)

</details>

## Contributing

We appreciate all contributions to improve MMClassification.
Please refer to [CONTRUBUTING.md](https://mmclassification.readthedocs.io/en/latest/community/CONTRIBUTING.html) for the contributing guideline.

## Acknowledgement

We thank the MMClassification.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new classifiers.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{earthnets4eo,
    title={EarthNets: Empowering AI in Earth Observation},
    author={Zhitong Xiong, Fahong Zhang, Yi Wang, Yilei Shi, Xiao Xiang Zhu},
    journal = {arXiv:2210.04936},
    year={2022}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
