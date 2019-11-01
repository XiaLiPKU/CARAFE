# CARAFE
An unofficial implementation of [CARAFE: Content-Aware ReAssembly of FEatures](https://arxiv.org/abs/1905.02188)

## Usage

Download the raw file of carafe.py into your project, and then import it by:
```from carafe import CARAFE```

## Some results

By now, I've only experimented on the Sementic Segmentation task. The results are reported on the Cityscapes dataset.
The backbone is ResNet-101 with output stride 32 (no dilation is adopted). For more details, please refer Table 6 in the original paper.
*PPM* and *FUSE* are not adopted here. I only compare upon *FPN* here.

|Methods 		| mIoU |
|:---------:|:----:|
|Bilinear		|74.52 |
|CARAFE(k=3)|78.16 |	
|CARAFE(k=5)|78.82 |
