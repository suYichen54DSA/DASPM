## DASPM: An Unsupervised Domain Adaptation Framework to Mapping Detailed Urban Land Use at Subpixel Scale

## Overview

The **urban system** is a complex hierarchical structure, rather than a simple linear model, 
within the domain of remote sensing. Taking urban road network planning as an example, 
it encompasses multiple scales of transportation networks ranging from sidewalks to highways,
 rather than just single lanes. The essence of this hierarchical structure pertains to the issue of scale,
  wherein mapping and surveying results exhibit significant differences under different observation scales (Mandelbrot, 1967).

Traditional remote sensing semantic segmentation typically employs a one-to-one correspondence between observation scales,
 such as pairing 10-meter spatial resolution imagery with 10-meter resolution land cover annotations. 
 However, the objective of **super-resolution mapping** is to acquire high-resolution land use mapping results 
 at smaller scales from low-resolution imagery at larger scales, thereby enhancing the visualization effect of mapping.

However, the ground reference data relied upon in super-resolution mapping often pertains solely to the land cover 
classification system corresponding to the resolution of the low-resolution imagery. 
This results in the loss of **semantic information** from large to small scales.
 For instance, in 30-meter resolution Landsat imagery, only impervious surfaces and non-impervious surfaces may be distinguished, 
 whereas in 10-meter resolution Sentinel-2 imagery, more detailed categories such as roads, buildings, and parking lots can be identified.

This phenomenon reflects the spatial heterogeneity and non-linear (long-tail distribution) complexity of 
spatial semantic information in different observation scales (Jiang, 2015): 
namely, smaller (detail) categories are far more numerous than larger (coarse) categories. Therefore, 
the aim of this study is to address this issue.

The **The Domain Adaptation-based Specific Sub-pixel Mapping(DASPM)** framework, based on the teacher-student model, 
aligns the target domain branch with the source domain branch in the remote sensing and machine learning domains, 
thereby complementing the mapping results of low-resolution images with high-resolution semantic information. 
It employs modules such as category enhancement and cross-domain fusion to address the impact of class imbalance, 
thus enhancing model generalization.




If you find this project useful in your research, please consider citing:
<!-- 
```

``` -->

## Comparison with Other UDA model

Currently, Unsupervised Domain Adaptation (UDA) method has not been widely applied in the field of remote sensing, resulting in a lack of remote sensing datasets for evaluating UDA frameworks. In order to facilitate better comparisons with other land mapping methods and to expand the categories for domain transfer, this study also utilizes a combination of 2-meter resolution high-resolution imagery obtained from Google Earth and 10-meter resolution Sentinel-2 imagery to create a land use dataset for comparative experiments. The link to this dataset is provided below:

Baidu Netdisk link:    https://pan.baidu.com/s/1U4L_h8G3rGjmRhRBb-rydg?pwd=A408
Extraction code:       A408
## Comparison with Other UDA model

DASPM significantly outperforms previous works on several UDA benchmarks in the dataset that we utilized.

![Alt text](image.png)


References:


## Setup Environment

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/DASPM
source ~/venv/DASPM/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Further, please download the MiT weights and a pretrained DASPM using the
following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```

All experiments were executed on a NVIDIA RTX 2080 Ti.

## Inference Demo

Already as this point, the provided DASPM model (downloaded by
`tools/download_checkpoints.sh`) can be applied to a demo image:

```shell
python -m demo.image_demo demo/demo.png work_dirs/211108_1622_gaofen2sentinel_daformer_s0_7f24c/211108_1622_gaofen2sentinel_daformer_s0_7f24c.json work_dirs/211108_1622_gaofen2sentinel_daformer_s0_7f24c/latest.pth
```


## Setup Datasets

Please download the dataset we have created and place it in the root directory of DASPM. The storage paths are as follows:
**data/gaofen** and **data/sentinel**.



The final folder structure should look like this:

```none
DASPM
├── ...
├── data
│   ├── gaofen
│   │   ├── image
│   │   ├── label
│   ├── sentinel
│   │   ├── image
│   │   ├── label
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gaofen2m.py data/gaofen --nproc 8
python tools/convert_datasets/sentinel10m.py data/sentine; --nproc 8
python tools/convert_datasets/global23k.py /home/heda/documents/Dataset/Dataset_6300_211/experiment1/ --nproc 8
```

## Training
To speed up the training process, you can use the pre-trained models provided by DAFormer originally [annotated config file](configs/DASPM/gaofen2sentinel_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py) 
A training job can be launched using:

```shell
python run_experiments.py --config configs/DASPM/gaofen2sentinel_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py
```



## Testing & Predictions


Your models can be tested after the training has finished:

```shell
sh test.sh path/to/checkpoint_directory
```
Typically, the checkpoint saved in '**work_dirs**' by default:


## Checkpoints

Below, we provide checkpoints of DASPM for different benchmarks.


<!-- * [DASPM for GTA→Cityscapes](https://drive.google.com/file/d/1pG3kDClZDGwp1vSTEXmTchkGHmnLQNdP/view?usp=sharing) -->

The checkpoints come with the training logs. Please note that:



## Framework Structure





## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [DAformer](https://github.com/lhoyer/DAFormer)

## License

