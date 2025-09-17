# Post-Hoc Explainability for Deep Learning

This project implements three post-hoc explainability methods — **DeconvNet**, **Grad-CAM**, and **Occlusion Sensitivity** — to interpret Convolutional Neural Networks (CNNs) trained on medical imaging datasets.  
It adapts **AlexNet** to handle **128×128 MedMNIST datasets** (PathMNIST, BloodMNIST, DermaMNIST), providing clear visual insights into how the model makes decisions.  
> **Note:** This work was developed as a group project at Télécom Paris.

---

## 📝 Project Overview
Deep learning models often achieve high performance in medical imaging but lack transparency. In this project, we address this challenge by applying state-of-the-art post-hoc explainability techniques to CNNs.  
We adapted AlexNet to the MedMNIST datasets and implemented visualization tools to understand which regions and features drive classification decisions.  

---

## 🔍 Key Features
- **AlexNet adaptation** for 128×128 medical images.  
- **DeconvNet**: reconstructs activations back into image space.  
- **Grad-CAM**: generates gradient-based heatmaps highlighting relevant regions.  
- **Occlusion Sensitivity**: evaluates the impact of masking image regions on predictions.  
- High classification performance on multiple medical imaging benchmarks.  

---

## 📊 Results
- **PathMNIST:** ~91% test accuracy  
- **BloodMNIST:** ~97% test accuracy  
- **DermaMNIST:** ~77% test accuracy  
- Visualizations clearly identify the image regions most influential to CNN predictions, improving model interpretability and trust.  

---

## 👩‍💻 Contributors
This project was carried out by a team of students at Télécom Paris:  
- MOREIRA TEIXEIRA Luiz Fernando 
- SANGINETO JUCÁ Marina 
- MARTINS Lorenza  
- YURI Franz

---

## 📄 Report
For full details on methodology, experiments, and results, please see the attached report (PDF).
