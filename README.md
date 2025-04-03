# Bug Detection Using Deep Learning

## Overview
This project implements a bug detection system using Natural Language Processing (NLP) and Deep Learning. The system is designed to classify code snippets based on potential bug types. It utilizes an LSTM-based neural network to analyze code and predict the type of bug present.

## Features
- Preprocessing of code snippets using tokenization and padding
- Classification of code bugs using an LSTM model
- One-hot encoding for label classification
- Model training and evaluation
- Ability to predict bug types for new code snippets
- Model saving and reloading for future use

## Dataset
The dataset used for training contains code snippets labeled with their respective bug types. The dataset is loaded from `sample_bug_dataset.csv`, which includes two columns:
- `code_snippet`: The code sample to analyze
- `bug_type`: The corresponding bug classification

## Requirements
Before running the project, ensure you have the following dependencies installed:
```bash
pip install tensorflow scikit-learn pandas numpy pickle
```

## Project Structure
```
├── bug_detector_model.h5    # Trained LSTM model
├── tokenizer.pkl            # Tokenizer object for text processing
├── labels.pkl               # Label mapping dictionary
├── sample_bug_dataset.csv   # Dataset used for training
├── bug_detection.py         # Main script containing the model and functions
├── README.md                # Project documentation
```

## Implementation
### 1. Data Preprocessing
- The dataset is loaded using pandas.
- Text data is tokenized and converted into sequences.
- Sequences are padded to ensure uniform input size.
- Labels are converted into numerical values and one-hot encoded.
- Data is split into training and testing sets.

### 2. Model Architecture
The model consists of the following layers:
- **Embedding Layer**: Converts tokenized input into dense word embeddings.
- **LSTM Layer**: Captures sequential dependencies in the code snippets.
- **Dense Layers**: Process the extracted features and classify bug types.
- **Softmax Activation**: Provides a probability distribution over bug types.

### 3. Training and Evaluation
- The model is trained using the categorical crossentropy loss function and the Adam optimizer.
- Model accuracy is measured to evaluate performance.
- The trained model is saved for future use.

### 4. Bug Type Prediction
A function `predict_bug_type(code_snippet)` is provided to analyze a new code snippet and predict the type of bug.

## How to Run
1. Train the model:
```python
python bug_detection.py
```
2. Predict bug type:
```python
new_code = "x = 5
if x = 10: print('x is 10')"
predicted_bug = predict_bug_type(new_code)
print(f"Predicted Bug Type: {predicted_bug}")
```

## Model Saving and Loading
The trained model and tokenization objects are saved for reuse:
```python
model.save("bug_detector_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)
```

To load the model and tokenizer:
```python
from tensorflow.keras.models import load_model
import pickle

model = load_model("bug_detector_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)
```

## Results
The model achieves an accuracy of approximately **XX%** (replace with actual accuracy after training).

## Future Improvements
- Expand dataset with more diverse bug types.
- Implement advanced NLP techniques (e.g., transformers) for better accuracy.
- Improve model explainability with attention mechanisms.
- Deploy the model as a web API for real-world usage.

## Author
Developed as part of a project on bug detection using deep learning.

## License
This project is released under the MIT License.
