# 🧠 Waste Segregation using Transfer Learning

This project applies **Transfer Learning** techniques to automate the **classification of nine types of waste** using **deep learning models**. The project explores how pre-trained **Convolutional Neural Networks (CNNs)** can be fine-tuned to achieve high performance on limited data.

---

## 🚀 Project Overview

The goal of this project is to build an efficient **multi-class image classifier** capable of identifying different categories of waste such as **glass, plastic, metal, paper, and organic materials**.  
By leveraging **Transfer Learning** from pre-trained models, the project demonstrates how to adapt existing architectures to domain-specific datasets for **environmental sustainability and smart waste management**.

---

## 🧩 Key Features

- **Transfer Learning Models:** ResNet50, VGG16, and EfficientNetB0
- **Frameworks Used:** Keras, TensorFlow, NumPy, Matplotlib, OpenCV
- **Techniques Applied:**
  - Data preprocessing (resizing, normalization, one-hot encoding)
  - Data augmentation (rotation, zoom, shift, flip)
  - Model fine-tuning and regularization
  - Training-validation split (80/20)
- **Evaluation Metrics:** Precision, Recall, AUC, F1-score
- **Result:** Achieved up to **25% improvement in validation accuracy** through model fine-tuning and augmentation

---

## 📊 Dataset

- Each class folder contains labeled images of waste types.  
- 80% of images are used for training, and 20% for testing.  
- Images are resized and normalized for model input consistency.

---

## 🧠 Model Training Workflow

1. **Data Preprocessing:**  
   - Loaded and resized all images to a consistent shape.  
   - Applied one-hot encoding for categorical labels.  

2. **Transfer Learning Setup:**  
   - Imported pre-trained CNN architectures (ResNet50, VGG16, EfficientNetB0).  
   - Replaced final layers for 9-class classification.  
   - Used Adam optimizer and categorical cross-entropy loss.

3. **Model Evaluation:**  
   - Compared model performance across training, validation, and test sets.  
   - Evaluated Precision, Recall, AUC, and F1-score.  

4. **Visualization:**  
   - Plotted accuracy/loss curves and confusion matrix for best model.  

---

## 📈 Results

| Model         | Precision | Recall | F1-Score | AUC  |
|----------------|------------|---------|----------|------|
| ResNet50       | 0.91       | 0.88    | 0.89     | 0.94 |
| EfficientNetB0 | 0.93       | 0.90    | 0.91     | 0.96 |
| VGG16          | 0.87       | 0.85    | 0.86     | 0.92 |

✅ **Best Model:** EfficientNetB0 — highest F1-score and AUC with balanced performance.

---

## 🧰 Tech Stack

- **Languages:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, Pandas, Matplotlib, OpenCV  
- **Environment:** Google Colab / Jupyter Notebook  
- **Version Control:** Git, GitHub

---

## 💡 Future Improvements

- Add support for **real-time waste detection** using OpenCV or YOLOv8.  
- Extend dataset for **global waste categories**.  
- Deploy model using **Streamlit** or **Flask** for user-friendly interaction.  

---

## 👩‍💻 Author

**Maitrri Chandra**  
📍 USC Data Science | AI & Analytics Researcher  
🔗 [Portfolio](https://maitrrichandra.framer.website/) • [LinkedIn](https://www.linkedin.com/in/maitrrichandra/) • [GitHub](https://github.com/Maitrri)

---

## 🏆 Acknowledgements

Project completed as part of **DSCI 552 - Machine Learning for Data Science** under the guidance of **Dr. M. R. Rajati, University of Southern California (USC).**

---

## 📄 License

This project is licensed under the MIT License - feel free to use or modify with attribution.

---

⭐ **If you found this project helpful, please star the repo!**
