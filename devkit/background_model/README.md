# To Use This Atmospheric Density Forecasting Toolbox

Ensure you've downloaded the data via the data link below. The files ending in r100.mat are pre-trained reduced-order spatial models. You may also use the density.mat files to train your own reduced-order spatial model using the code in the reduced-order model generation folder. If you want to generate more NRLMSISE data, an example file for running the NRLMSISE model is included in the generate density data file. The following steps allow you to train your own Transformer or Dynamic Mode Decomposition with control (DMDc)-based forecasting model for atmospheric density.

1. Load, standardize, and determine data splits using DataGenerationandStandardization.ipynb. The current example shows how to split data based on solar activity, including high, medium, and low levels of solar activity.
2. Train your Transformer Neural Network with TrainingandValidation.ipynb. If using DMDc, you do not need this step.
3. Use EvaluationTransformer.ipynb and EvaluationDMDc.ipynb to evaluate your trained model's performance on you evaluation data splits.

Data Link: https://drive.google.com/drive/folders/1IHFrqF-lnvAeAOVdvSV288sByYX9I5HB?usp=sharing

The data preprocessing, training, and evaluation notebooks are adapted and modified from [PatchTST](https://github.com/yuqinie98/PatchTST) [^1].

Research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

## Citing

If you use this code in your work, kindly cite the following associated publication.

```
@article{Briden2023,
  year = {2023},
  month = Sept,
  author = {Julia Briden and Peng Mun Siew and Victor Rodriguez-Fernandez and Richard Linares},
  title = {Transformer-based Atmospheric Density Forecasting},
  journal = {Advanced Maui Optical and Space Surveillance (AMOS) Technologies Conference},
  note = {Free preprint available at [https://arxiv.org/abs/2310.16912](https://arxiv.org/abs/2310.16912)}
}
```
[^1]: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." International Conference on Learning Representations, 2023.
