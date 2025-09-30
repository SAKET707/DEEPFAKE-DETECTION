# Deepfake Classifier

A deep learning–based image classifier to distinguish **real** vs **fake (deepfake)** images.
Built with **PyTorch**, this model achieves **98.70% accuracy** on a balanced dataset of **8,000 images** (4k real + 4k fake).

---
## Sreamlit APP
* URL : https://deepfake-detection-by-saket.streamlit.app/
---

## 🚀 Features

* **Custom CNN architecture** implemented in PyTorch
* Input preprocessing with `torchvision.transforms`
* Trained on balanced dataset with two classes:

  * `0 → Fake`
  * `1 → Real`
* Achieved **98.70% accuracy** on validation set

---
## 🚀 Demo
|           |           |
|-----------|-----------|
| <img width="400" height="500" alt="eg1 fake" src="https://github.com/user-attachments/assets/64a7d09f-b56a-4960-8296-53d37a72aee6" /> | <img width="400" height="500" alt="eg1 real" src="https://github.com/user-attachments/assets/b595f352-6e11-480f-b4ae-34811f182e6e" />
 |

---


## 🧠 Model Architecture

```python
class DeepfakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(64*28*28,512),
            nn.ReLU(),
            nn.Linear(512,2)
        )
    def forward(self,x):
        return self.network(x)
```

---

## 🛠️ Preprocessing

Every input image is resized, converted to tensor, and normalized before inference:

```python
transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
```

---

## 📊 Results

* **Training Dataset:** 8,000 images (4k real, 4k fake)
* **Testing Dataset:** 2,000 images (1k real, 1k fake)
* **Accuracy:** 98.70%

---

## ⚖️ Disclaimer

This project is for **educational purposes only**.
It is not intended for harmful use or to mislabel content outside research/academic scope.

---
