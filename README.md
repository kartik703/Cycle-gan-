# CycleGAN for Image Style Transfer

## Overview
This project implements a **CycleGAN** model to perform image-to-image translation between two domains: **human faces and cats/dogs**. The goal is to learn a mapping that translates images between these domains without paired data.

## Features
- Unpaired image-to-image translation using **CycleGAN**.
- Training on datasets of human faces and pet images (cats/dogs).
- Evaluation of model performance with loss metrics and qualitative results.
- Implementation in **Python** using **PyTorch**.
- Preprocessing steps including data augmentation and normalization.

## Dataset
The project requires datasets containing:
- **Human Faces:** Images of human faces.
- **Cats/Dogs:** Images of cats and dogs.

The datasets are preprocessed and loaded using **PyTorch's DataLoader**.

## Training
To train the CycleGAN model, run the provided Jupyter Notebook **Cycle_gan.ipynb**:
1. Load the datasets.
2. Preprocess images (resize, normalize, augment).
3. Train the model using adversarial and cycle consistency losses.
4. Save checkpoints for inference.

## Evaluation & Inference
- Use the trained model to generate translated images.
- Compare generated results qualitatively and quantitatively.
- Evaluate using cycle consistency loss and adversarial loss.

## Results
- The generated images should exhibit realistic transformations between human faces and pets.
- Visualizations of the transformations will be stored in the `results/` folder.

## Future Work
- Fine-tune hyperparameters for better results.
- Experiment with different architectures and loss functions.
- Apply CycleGAN to other domains beyond faces and pets.

## References
- [Unpaired Image-to-Image Translation using CycleGAN (Paper)](https://arxiv.org/abs/1703.10593)
- [PyTorch Documentation](https://pytorch.org/)

