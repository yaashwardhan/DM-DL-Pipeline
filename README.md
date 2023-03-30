## 7 Tasks Completed :mag::white_check_mark:

[:arrow_right: Click Here :arrow_left:](https://drive.google.com/drive/folders/1x5gm4ywOQ8brxMn11KjLBxsehQuxtx19?usp=sharing) to access all the data including the trained models for all tasks.

---
### Results: 
|   |   |   |
|---|---|---|
|||<img src="Task1 - MultiLabel Classification (0.98 AUC) (AttentionCNN, ViT, Resnet50)/results/results_Channelwise_Attention_CNN.png" width="600">
|||<img src="Task1 - MultiLabel Classification (0.98 AUC) (AttentionCNN, ViT, Resnet50)/results/ResNet50_results.png" width="600">   
|||<img src="Task1 - MultiLabel Classification (0.98 AUC) (AttentionCNN, ViT, Resnet50)/results/Custom_ViT.png" width="600">


---
About: 
Task1 - MultiLabel Classification (0.98 AUC) (AttentionCNN, ViT, Resnet50)

I have trained 4 different models from scratch for this task in the notebook inside the Task 2 folder. I also evaluated then on the 10% Validation Data for MSE (Mean Squared Error), SSIM (Structural Similarity Index) and PSNR (Peak Signal-to-Noise Ratio).

The custom trained models and their evaluation on 10% val set are: 

| Model      | MSE        | SSIM       | PSNR       |
|------------|------------|------------|------------|
| SuperResCNN (Super-Resolution Convolutional Neural Network) | 0.000065   | 0.99168    | 41.780569  |
| EDSR (Enhanced Deep Residual Networks)       | 0.000298   | 0.987563   | 36.769835  |
| LapSRN (Laplacian Pyramid Super-Resolution Network)     | 0.004762   | 0.509009   | 22.244892  |
| ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)     | 0.000968   | 0.967625   | 27.573939  |

The code and all functions are well documented in the ipynb file. 



<a href="https://tensorflow.org"><img src="https://img.shields.io/badge/Powered%20by-Tensorflow-orange.svg"/></a> [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) ![GitHub last commit (branch)](https://img.shields.io/github/last-commit/yaashwardhan/BrainStain.ai/main?color=blue)
