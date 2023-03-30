## 7 Tasks Completed :mag::white_check_mark:

[:arrow_right: Click Here :arrow_left:](https://drive.google.com/drive/folders/1x5gm4ywOQ8brxMn11KjLBxsehQuxtx19?usp=sharing) to access all the data including the trained models for all tasks. 

Everything is built in Keras and Tensorflow. If required, I can do the same in PyTorch aswell.

I am interested in applying for all the proposals/projects, especially:
- `Superresolution for Strong Gravitational Lensing`
- `Self-Supervised Learning for Strong Gravitational Lensing`
- `Transformers for Dark Matter Morphology with Strong Gravitational Lensing`

---
### Details and results for all tasks: 

*   **Task1:** Multi-Label Classification(Results get better over time)

| Approaches | Val AUC  | Confusion Matrix and ROC plot  |
|---|---|---|
|*`Channelwise Attention CNN`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task1%20-%20MultiLabel%20Classification%20(0.98%20AUC)%20(AttentionCNN%2C%20ViT%2C%20Resnet50)/Task1_Channelwise_Attention_CNN.ipynb)<br>This approach involves using<br>a CNN with two branches, each<br>containing a channelwise attention<br>mechanism to refine learned features. |0.80|<img src="Task1 - MultiLabel Classification (0.98 AUC) (AttentionCNN, ViT, Resnet50)/results/results_Channelwise_Attention_CNN.png" width="600">
|*`Vision Transformer (Custom)`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task1%20-%20MultiLabel%20Classification%20(0.98%20AUC)%20(AttentionCNN%2C%20ViT%2C%20Resnet50)/Task1_ViT_from_scratch.ipynb)<br>This approach involves<br>a self-attention Vision Transformer<br>whoes architecture implemented from<br>scratch and then imagenet<br>pretrained weights are applied to it.<br>The model processes image patches<br>through 12 transformer blocks with<br>multi-head self-attention and MLP,<br>then outputs class probabilities. |0.90|<img src="Task1 - MultiLabel Classification (0.98 AUC) (AttentionCNN, ViT, Resnet50)/results/Custom_ViT.png" width="600">
|*`ResNet50 Transfer Learning`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task1%20-%20MultiLabel%20Classification%20(0.98%20AUC)%20(AttentionCNN%2C%20ViT%2C%20Resnet50)/Task1_ResNet50_transfer.ipynb)<br>Utilizing ResNet-50 for transfer<br>learning, we remove its classification<br>head, apply batch normalization,<br>dropout, and a dense<br>layer with softmax activation<br>for 3-class probability output.<br>This implementation is simplified<br>using existing libraries for<br>the model's architecture.|0.98| <img src="Task1 - MultiLabel Classification (0.98 AUC) (AttentionCNN, ViT, Resnet50)/results/ResNet50_results.png" width="600">

---

*   **Task2:** Lens Finding

| Approach | Val AUC  | Confusion Matrix and ROC plot  |
|---|---|---|
|*`Self-Attention-CNNs`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task2%20-%20Lens%20Finding%20(0.99%20AUC)%20(Self-Attention%20CNN)/Task2_Lens_Finding_Self_Attention_CNN.ipynb)<br>A multimodal model using CNNs<br>and attention mechanisms to process<br>images and features.<br>The model combines the image and <br>feature branches, applies self<br>attention,and outputs a probability<br>through Dense layers. | 0.99 |<img src="Task2 - Lens Finding (0.99 AUC) (Self-Attention CNN)/lens_finding_results.png" width="600">

---

*   **Task3:** Learning Mass of Dark Matter Halo

| Approach | MSE | 
|---|---|
|*`Representational Learning Transformers`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task3%20-%20Learning%20Mass%20of%20Dark%20Matter%20Halo%20%20(2.28x10%5E-4%20MSE)%20(Equivariant%20Transformers)/Learning_Mass_of_Dark_Matter_Halo_Regressor.ipynb.ipynb)<br>Transformers use custom RotationalConv2D layers and contrastive loss<br>to learn equivariant representations, improving performance on tasks involving<br>image augmentations like rotations. The model is pre-trained with ResNet50<br>weights and fine-tuned for specific regression tasks. | 2.28 x 10^-4

---

*   **Task4:** Exploring Equivariant Neural Networks

| Approach | Val AUC  | Confusion Matrix and ROC plot  |
|---|---|---|
|*`Self Supervised Equivariant Transformers`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task4%20-%20Equivariant%20Neural%20Networks%20(0.99%20AUC)/Task4_Classification_Equivariant_Transformer.ipynb)<br>Equivariant Transformers use custom<br>RotationalConv2D layers and ResNet50<br>transfer learning to maintain<br>equivariance for input rotations.<br>Contrastive loss guides embeddings,<br>followed by fine-tuning for classification tasks.  | 0.99 |<img src="Task4 - Equivariant Neural Networks (0.99 AUC)/equivariant_transformers_classification_results.png" width="600">

---

*   **Task5:** Exploring Vision Transformers

| Approach | Val AUC  | Confusion Matrix and ROC plot  |
|---|---|---|
|*`Vision Transformers`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task5%20-%20Vision%20Transformers%20(0.99%20AUC)%20From%20Scratch/ViT_from_scratch_Task5.ipynb)<br>(Self-Written, inspired by vit-keras<br>which is not maintained since 2021).<br>Uses self-attention mechanisms.<br> We follow detailed steps,<br>including 2D Conv layer,<br>token flattening, positional embeddings,<br>and transformer blocks, to implement<br>the model and apply pretrained 'npz'<br>weights for prediction.  | 0.99 |<img src="Task5 - Vision Transformers (0.99 AUC) From Scratch/ViT_results.png" width="600">

---

*   **Task6:** Image SuperResolution

| Approach      | MSE        | SSIM       | PSNR       |
|------------|------------|------------|------------|
| *`SuperResCNN`* (Super-Resolution Convolutional Neural Network)<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task6%20-%20Image%20Super-resolution%20(0.99%20SSIM%2C%2041.7%20PSNR)%20(SuperResCNN%2C%20EDSR%2C%20LapSRN%2C%20ESRGAN)/Task6_SuperResolution_Yashwardhan.ipynb)<br>Establish a baseline model for performance analysis to guide improvement direction (e.g., residual blocks, self-attention, or GAN architecture). Begin with SuperResCNN, an upsampling layer and three-layer neural network for mapping low-resolution to high-resolution images. | 0.000065   | 0.99168    | 41.780569  |
| *`EDSR`*  (Enhanced Deep Residual Networks)<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task6%20-%20Image%20Super-resolution%20(0.99%20SSIM%2C%2041.7%20PSNR)%20(SuperResCNN%2C%20EDSR%2C%20LapSRN%2C%20ESRGAN)/Task6_SuperResolution_Yashwardhan.ipynb)<br>Residual Blocks to capture more complex image features       | 0.000298   | 0.987563   | 36.769835  |
| *`LapSRN`*  (Laplacian Pyramid Super-Resolution Network)<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task6%20-%20Image%20Super-resolution%20(0.99%20SSIM%2C%2041.7%20PSNR)%20(SuperResCNN%2C%20EDSR%2C%20LapSRN%2C%20ESRGAN)/Task6_SuperResolution_Yashwardhan.ipynb)<br>LapSRN, preserves details with an Add() layer in the residual_block function, improving memory efficiency and speeding up inference, while reducing blur and sharpening the image.     | 0.004762    | 0.509009   | 22.244892  |
| *`ESRGAN`* (Enhanced Super-Resolution Generative Adversarial Networks) <br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task6%20-%20Image%20Super-resolution%20(0.99%20SSIM%2C%2041.7%20PSNR)%20(SuperResCNN%2C%20EDSR%2C%20LapSRN%2C%20ESRGAN)/Task6_SuperResolution_Yashwardhan.ipynb)<br>Generative Adversarial Networks can combat mode collapse using loss functions like perceptual loss, which leverages VGG19 and sub-pixel convolution for high-resolution image generation. Residual Dense Blocks, batch normalization, and other techniques help stabilize and improve training for visually accurate results.    | 0.000968   | 0.967625   | 27.573939  |

<img src="Task6 - Image Super-resolution (0.99 SSIM, 41.7 PSNR) (SuperResCNN, EDSR, LapSRN, ESRGAN)/model_results.jpg">

---

*   **Task8:** Self-Supervised Learning 
 
| Approaches | Metrics  | Confusion Matrix and ROC plot  |
|---|---|---|
|*`Classification-Self_Supervised`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/f46a5783742973762c5bfd189b01a99f78a397ed/Task8%20-%20Self-Supervised%20Learning%20(0.99%20AUC,%202.28x10%5E-4%20MSE)%20(Representational%20Learning,%20Equivariant%20Transformers,%20Constrastive%20Loss)/Task8_Classification_SelfSupervised_Equivariant_Transformer.ipynb)<br>Equivariant Transformers use custom<br>RotationalConv2D layers and ResNet50<br>transfer learning to maintain<br>equivariance for input rotations.<br>Contrastive loss guides embeddings,<br>followed by fine-tuning for classification tasks. |0.99 AUC |<img src="Task8 - Self-Supervised Learning (0.99 AUC, 2.28x10^-4 MSE) (Representational Learning, Equivariant Transformers, Constrastive Loss)/results/self_supervised_equivariant_transformers_classification_results.png" width="600">|
|*`Regression-Self_Supervised`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/Evaluation-Test-DeepLense/blob/main/Task8%20-%20Self-Supervised%20Learning%20(0.99%20AUC%2C%202.28x10%5E-4%20MSE)%20(Representational%20Learning%2C%20Equivariant%20Transformers%2C%20Constrastive%20Loss)/Task8_Regression_SelfSupervised_Transformer_Representational_Learning.ipynb)<br>Transformers use custom RotationalConv2D layers and contrastive loss<br>to learn equivariant representations, improving performance on tasks involving<br>image augmentations like rotations. The model is pre-trained with ResNet50<br>weights and fine-tuned for specific regression tasks.|2.28 x 10^-4 MSE|
