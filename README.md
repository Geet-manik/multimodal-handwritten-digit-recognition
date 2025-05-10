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


The script will:
- Download both datasets automatically.
- Train the multimodal model.
- Print training loss and test accuracy.
- Show visualizations of predictions.

---

## 📊 Results

- The multimodal model achieves high accuracy on the test set.
- Combining image and pen stroke features improves robustness, especially for ambiguous digits.
- The script also displays a visualization of sample predictions.

---

## 📝 Reflections

- **What worked:**  
- The model effectively fuses information from both modalities.
- Pen stroke features help clarify ambiguous images.
- **Limitations:**  
- Datasets are not perfectly aligned (not the same digit instances), but this approach demonstrates the concept.
- **Future Work:**  
- Use datasets with paired image and pen stroke data (e.g., Quick, Draw!).
- Experiment with more advanced fusion techniques (attention, late fusion).
- Try deeper neural networks for each branch.

---

## 📖 References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [UCI Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/ml/datasets/pen-based+recognition+of+handwritten+digits)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- DA623 Course Materials

---

**Project for DA623 Winter 2025 | Geetanjay Maink (210102033)**






