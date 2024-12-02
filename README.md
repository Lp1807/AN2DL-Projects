# ANN2DL Projects
- Ernesto Natuzzi
- Flavia Nicotri
- Luca Pagano
- Giuseppe Vitello
# First Homework
This repository contains the implementation and experimentation details of the Blood Cell Classification project, a task designed as part of the Artificial Neural Networks and Deep Learning course at Politecnico di Milano. The goal is to classify 96x96 RGB images of blood cells into eight categories, each representing a specific cell type, utilizing advanced augmentation techniques and deep learning models.

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
## More Information

For a detailed overview of the challenge, methods, and models built, please refer to the [report](Homework_1_Report.pdf) and the [notebooks](/Homework_1_Notebooks).

## Results:
Achieved a final test accuracy of 92% using an ensemble of EfficientNetV2S models with optimized augmentation and hyperparameters.
