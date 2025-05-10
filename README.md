# Multimodal Handwritten Digit Recognition

This project demonstrates a multimodal deep learning approach for handwritten digit recognition using **image data** (from the MNIST dataset) and **pen stroke features** (from the UCI Pen-Based dataset). The model combines both modalities to improve recognition accuracy.

---

## 📚 Motivation

Handwritten digit recognition is a classic machine learning problem. Traditionally, models use only image data (like MNIST). However, with the rise of touch devices and digital pens, pen stroke data (how the digit was written) is often available. Multimodal learning-using more than one type of input-can improve robustness and accuracy. This project explores how combining image and pen stroke data can enhance digit classification.

---

## 🗂️ Datasets

- **MNIST**: 70,000 grayscale images of handwritten digits (0–9), 28x28 pixels.  
  [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- **UCI Pen-Based Recognition of Handwritten Digits**: Pen stroke features for digits 0–9.  
  [UCI Pen-Based Dataset](https://archive.ics.uci.edu/ml/datasets/pen-based+recognition+of+handwritten+digits)

---

## 🏗️ Model Architecture

- **Image Branch**: Convolutional Neural Network (CNN) processes image data.
- **Pen Stroke Branch**: Fully connected layers process pen stroke features.
- **Fusion**: Concatenates outputs from both branches.
- **Classifier**: Fully connected layers output digit class probabilities.

**Diagram:**


