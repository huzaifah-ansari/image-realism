```markdown
# 🧠 AI vs Real Image Classifier

This project is a deep learning pipeline to classify whether an image is **AI-generated** or **Real**. It includes model training, evaluation, and a GUI-based image classifier using Tkinter.

---

## 🧩 Features

- Binary classifier trained on real vs AI-generated images
- Evaluation with confusion matrix, accuracy/loss/precision/recall plots
- GUI app using Tkinter to load and classify new images
- Confidence score displayed using a custom progress meter

---

## 🗂️ Project Structure



.
├── model/
│   ├── model\_train\_code.py        # Model training code
│   ├── best\_model.h5              # Saved best model
│   └── final\_model.h5             # Final trained model
│
├── gui/
│   └── implementation\_code.py     # GUI app using Tkinter
│
├── evaluation\_plots/
│   ├── confusion\_matrix.png
│   ├── training\_history.png
│   ├── evaluation\_results.txt
│   └── model\_architecture.txt
│
├── requirements.txt
└── README.md

````

---

## 🧠 Model Info

- Input size: 32×32 RGB images  
- Architecture: 3 Convolutional layers + Dense layers with dropout  
- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Metrics: Accuracy, Precision, Recall

---

## 📊 Evaluation Results

- Accuracy: ~95%
- Precision/Recall: High scores for both classes
- Early stopping and model checkpointing applied

### Confusion Matrix:

![Confusion Matrix](evaluation_plots/confusion_matrix.png)

### Training History:

![Training History](evaluation_plots/training_history.png)

---

## 🚀 Run the GUI App

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
````

Or manually:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow
```

### Step 2: Run the app

```bash
cd gui
python implementation_code.py
```

* Upload any image
* See the original + 32x32 processed version
* Get prediction: **Likely Real**, **Uncertain**, or **Likely AI-Generated**
* Confidence visualized using a progress meter

---

## 🧪 Retrain the Model (Optional)

1. Place your dataset under:
   `/home//Image Realism/archive/train/`
   With this structure:

   ```
   train/
   ├── ai/
   └── real/
   ```

2. Run the training script:

```bash
cd model
python model_train_code.py
```

---

## 📜 License

MIT License

---

## ✍️ Author

**huzaifah Ansari**
Real vs AI Image Detection
[GitHub](https://github.com/JaleedAhmad)
