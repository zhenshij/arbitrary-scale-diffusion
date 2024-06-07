# Arbitrary-Scale Image Generation and Upsampling using Latent Diffusion Model and Implicit Neural Decoder (CVPR 2024)

## Data preparation
Download the dataset you want to use and put it in the `../datasets/`
- [FFHQ](https://github.com/NVlabs/ffhq-dataset)
- [CelebaHQ](https://www.kaggle.com/badasstechie/celebahq-resized-256x256)
- [LSUN](https://github.com/fyu/lsun)
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)

Please place the txt file to split into training and validation images in the `/data`.
The split file for the lsun dataset can be downloaded from [here](https://ommer-lab.com/files/lsun.zip).

## Model Training
### Training First-Stage Models

```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/first-stage/<config_spec>.yaml -t --gpus 0, --scale_lr False
```

### Training LDMs
Creates or modifies the config file in `configs/latent-diffusion/`.
Type ckpt_path to load the first_stage_model.

```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t --gpus 0, --scale_lr False
```


## Test

### Super-Resolution
```
python eval_sr.py --exp logs/<exp_path> --lr_size <input_lr_image_size> --scale_ratio <scale>
```

### Image Generation
```
python inference.py --log_dir logs/<exp_path> --save_dir <output_path> --size <output_size_1> <output_size_2> ...
```
Measure the FID or SSIM between the real image and the generated image.

## Citation

If you find this work useful, please consider citing our paper.

```
@inproceedings{kim2024arbitraryscale,
      title={Arbitrary-Scale Image Generation and Upsampling using Latent Diffusion Model and Implicit Neural Decoder},
      author={Kim, Jinseok and Kim, Tae-Kyun},
      booktitle={CVPR},
      year={2024}
}
```