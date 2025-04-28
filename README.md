# Peptide-Membrane Interaction Analysis and Classification

### Overview

This project focuses on the **analysis of peptide interactions with hemispherical-planar lipid membrane architectures** using **molecular dynamics (MD) simulations** and the development of a **machine learning classifier** to predict peptide binding behavior. Specifically, it explores the factors influencing how peptides sense membrane curvature.

The workflow involves:
- **Analyzing MD trajectories** to calculate **biophysically relevant features** describing peptide insertion into membranes.
- **Preprocessing and feature engineering**, including **scaling**, **balancing data**, and **careful labeling**.
- **Building and evaluating a multi-layer perceptron (MLP) classifier** using scikit-learn to predict peptide binding states based on the extracted features.

