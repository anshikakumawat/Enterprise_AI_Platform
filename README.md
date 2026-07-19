# Enterprise AI Platform

An end-to-end AI platform that integrates Machine Learning, Deep Learning, Feature Engineering, Customer Analytics, and Forecasting into a single interactive Streamlit application.

## Features

### Neural Network Dashboard
- Perceptron implemented from scratch using NumPy
- Multi-Layer Perceptron (MLP)
- Configurable activation functions
  - Sigmoid
  - ReLU
  - Tanh
- Interactive training dashboard
- Accuracy and loss visualization
- Built-in datasets
- CSV upload support

---

### Sequence Models
- LSTM Model
- GRU Model
- Batch Normalization
- Dropout Regularization
- L2 Regularization
- TensorFlow/Keras implementation

---

### Feature Engineering
- Lag Feature Creation
- Target Encoding
- Column Transformer Pipeline
- Standard Scaling
- One-Hot Encoding
- SMOTE for Imbalanced Data Handling

---

### Anomaly Detection
- Isolation Forest
- Outlier Detection
- CSV Upload
- Prediction Visualization

---

### Customer Segmentation
- Customer clustering modules
- Data preprocessing
- Cluster visualization

---

### Demand Forecasting
- Time Series Forecasting
- Future demand prediction
- Forecast visualization

---

## Technologies Used

- Python
- Streamlit
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Imbalanced-learn

---

## Project Structure

```
Enterprise_AI_Platform/
│
├── app.py
│
├── customer_analytics/
│   ├── neural_nets.py
│   ├── sequence_models.py
│   ├── segmentation_engines.py
│   └── test_sequence.py
│
├── core_pipeline/
│   └── transformation_pipes.py
│
├── forecasting_engine/
│   └── time_series.py
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/anshikakumawat/Enterprise_AI_Platform.git
```

Move into the project

```bash
cd Enterprise_AI_Platform
```

Create a virtual environment

Windows

```bash
python -m venv .venv
```

Activate the virtual environment

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## Modules Included

- Neural Networks
- Sequence Models
- Feature Engineering
- Anomaly Detection
- Customer Segmentation
- Demand Forecasting

---

## Contributors

- Anshika Kumawat
- Khushi Rani
-Shivani Kumari
-samridhi Kumari
---

## Future Improvements

- Transformer Models
- Explainable AI (SHAP/LIME)
- Hyperparameter Optimization
- Model Deployment
- AutoML Integration
- Real-time Prediction APIs

---

## License

This project is developed for educational and research purposes.
