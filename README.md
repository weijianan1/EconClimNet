# EconClimNet
This repository contains the source code EconClimNet for our paper: "Climate Change from Carbon Dioxide Removal Benefits Global Economy: Insights from Deep Learning".

## Reproduction

1. Find a device with GPU support. Our experiment is conducted on a single RTX 24GB GPU and in the Linux system.

2. Install Python 3.8.18. The following script can be convenient.
```bash
pip install -r requirements.txt
```

3. Download the ERA5 dataset from [[Google Drive]](https://drive.google.com/file/d/1u4UQk0M_Ht3jKEKZGrmFdsvlHgAdXibh/view?usp=sharing). And place them under the './dataset/ERA5' folder.

4. Download the CMIP5 dataset from [[Google Drive]](https://drive.google.com/file/d/1b9d56N1abrYimzaDDUGzK5CA-t16As3R/view?usp=sharing). And place them under the './dataset/CMIP5' folder.

5. Train and evaluate the model with the following scripts.

```shell
bash ./scripts/train_gdp.sh
bash ./scripts/train_ag.sh
bash ./scripts/train_man.sh
bash ./scripts/train_serv.sh
```

```shell
bash ./scripts/eval_gdp.sh
bash ./scripts/eval_ag.sh
bash ./scripts/eval_man.sh
bash ./scripts/eval_serv.sh
```

6. We also provide the checkpoints for gdp, ag, man, ser settings in [[Google Drive]](https://drive.google.com/file/d/1O_jiN-8zxztd_wNJBQE4rm8Bo3V9hZCz/view?usp=sharing)

## Citation

If you find our work useful in your research, please consider citing:

```
@article{
  title={{Climate Change from Carbon Dioxide Removal Benefits Global Economy: Insights from Deep Learning}},
  author={Bin Tang and Jianan Wei and Anmin Duan and Yimin Liu and Bian He and Wen Bao and Wenguan Wang and Wenting Hu},
}
```

## Contact
If you have any questions or suggestions, feel free to contact Jianan Wei (weijianan.gm@gmail.com).




