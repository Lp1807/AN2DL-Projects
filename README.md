# Artificial Neural Network and Deep Learning Projects

Our "Alpaca" team was composed by:
- [Ernesto Natuzzi](https://github.com/Ernesto1717)
- [Flavia Nicotri](https://github.com/flanico)
- [Luca Pagano](https://github.com/Lp1807)
- [Giuseppe Vitello](https://github.com/Peppisparrow)
  
<img src="https://github.com/user-attachments/assets/8a6e3515-e375-4f12-8cc9-366b46f928f8" width="250">

# First Homework

This repository contains the implementation and experimentation details of the Blood Cell Classification project, a task designed as part of the Artificial Neural Networks and Deep Learning course at Politecnico di Milano. The goal is to classify 96x96 RGB images of blood cells into eight categories, each representing a specific cell type, utilizing advanced augmentation techniques and deep learning models.
<img width="1049" alt=" bloodcells_classification" src="https://github.com/user-attachments/assets/c5b0c58b-46ae-4b6e-b6fe-68e74965ca5a" />

## Features

- **Dataset Preprocessing:**
  - Cleaned dataset from outliers and duplicates using feature extraction and statistical analysis.
  - Final dataset: 11,951 images with balanced class distribution.

- **Data Augmentation:**
  - Progressive augmentation strategies (e.g., flip, rotation, zoom, and color transformations).
  - Advanced methods such as RandAugment, AugMix, GridMask, and RandomCutout.

- **Model Architecture:**
  - Transitioned from custom CNNs to pre-trained models like MobileNetV3Large and EfficientNetV2S.
  - Fine-tuning strategies to optimize generalization with heavily augmented datasets.

- **Techniques Explored:**
  - Class balancing with SMOTE (Synthetic Minority Over-sampling TEchnique).
  - Optimizers such as AdamW and experiments with alternatives like Lion and Ranger.
  - Incorporation of segmentation techniques for feature isolation.

## Results

Achieved a final test accuracy of 92% using an ensemble of EfficientNetV2S models with optimized augmentation and hyperparameters.

For a detailed overview of the challenge, methods, and models built, please refer to the [report](Homework_1_Report.pdf) and the [notebooks](/Homework_1_Notebooks).

# Second Homework

This repository contains the implementation and experimentation details of the Mars Terrain Segmentation project, a task designed as part of the Artificial Neural Networks and Deep Learning course at Politecnico di Milano. The objective is to segment 64x128 grayscale images of the Martian surface into pixel-wise masks with five categories: background, soil, bedrock, sand, and bigrock.
<img width="640" alt="Screenshot_2025_02_23_alle_174407_w_trans" src="https://github.com/user-attachments/assets/5500b508-3573-4870-940e-3a1b95a407e2" />

## Features

- **Dataset Preprocessing:**
  - Removal of outliers, specifically alien images identified through identical masks.
  - Final dataset: 2,505 labeled images with balanced distribution across classes.

- **Model Architecture and Evolution:**
  - Baseline established using a standard U-Net architecture.
  - Enhanced U-Net with residual blocks in all encoder and decoder layers, including the bottleneck.
  - Attention gates incorporated between symmetric upsampling and downsampling paths for better focus on critical regions.
  - Hybrid loss function combining CategoricalCrossEntropy and DiceLoss to optimize predictions.

- **Advanced Techniques:**
  - **Cut-and-Paste Augmentation:** Addressed class imbalance by augmenting "bigrock" patches into images. Discontinued due to limited improvements in test performance.
  - **Post-Processing:** Replaced background with the most frequent label and applied SciPy’s binary dilation to refine predictions.
  - **Double Enhanced U-Net:** Developed a two-stage U-Net configuration where the first model generates initial masks, and the second refines them using connections at input, bottleneck, and skip layers.
  - **Pseudo-Labeling:** Iteratively trained the second U-Net using predictions from unlabeled test data to enhance generalization.
  - **Test-Time Augmentation (TTA):** Improved predictions by applying horizontal and vertical flips during inference and averaging results.

## Results

- Significant performance improvement through iterative model refinements and innovative training strategies:
  - Baseline U-Net MeanIoU: 42.64%
  - Final Double Enhanced U-Net with TTA: 77.48%

- **Comparison Table:**

| Model/Configuration        | MeanIoU |
|----------------------------|---------|
| First Model               | 42.64%  |
| Augmented Model           | 55.11%  |
| Binary Dilation           | 67.09%  |
| First Double U-Net        | 68.45%  |
| Final Double U-Net        | 74.39%  |
| Pseudo-Labeling           | 76.10%  |
| TTA                       | 77.48%  |

## Conclusion

We achieved excellent results by leveraging all available data and incorporating advanced augmentation, architectural improvements, and post-processing techniques. Further enhancements could involve refining pseudo-labels using confidence thresholds to reduce noise and improve resilience.

For detailed experimentation, results, and methodology, please refer to the [report](Homework_2_Report.pdf) and the [notebooks](/Homework_2_Notebooks).
