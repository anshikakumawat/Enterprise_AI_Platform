import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from customer_analytics.neural_nets import Perceptron, MLP
from customer_analytics.segmentation_engines import render as render_clustering
from forecasting_engine.time_series import render as render_forecasting


st.set_page_config(page_title="Neural Network Dashboard", layout="wide")

st.title("Enterprise AI Platform")
st.subheader("Neural Network Dashboard")
st.caption("Perceptron & MLP implemented from scratch using NumPy")
tab_nn, tab_cluster, tab_forecast = st.tabs(
    ["Neural Network", "Customer Segmentation", "Demand Forecasting"]
)

with tab_nn:
    st.sidebar.header("Configuration")

    data_source = st.sidebar.radio("Dataset Source", ["Built-in Dataset", "Upload CSV"])

    if data_source == "Built-in Dataset":
        dataset_name = st.sidebar.selectbox("Dataset", ["Synthetic Binary", "Two Moons", "Iris (binary subset)"])
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV (last column = target)", type=["csv"])

    model_type = st.sidebar.selectbox("Model", ["Perceptron", "MLP"])
    activation_name = st.sidebar.selectbox("Activation", ["Sigmoid", "ReLU", "Tanh"])
    lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1, step=0.001)
    epochs = st.sidebar.slider("Epochs", 10, 500, 100, step=10)

    if model_type == "MLP":
        hidden_neurons = st.sidebar.slider("Hidden Neurons", 2, 64, 8)

    train_btn = st.sidebar.button("Train Model")


    def load_data():
        if data_source == "Built-in Dataset":
            if dataset_name == "Synthetic Binary":
                X, y = make_classification(n_samples=400, n_features=4, n_informative=3,
                                            n_redundant=0, random_state=42)
                cols = [f"feature_{i}" for i in range(X.shape[1])]
            elif dataset_name == "Two Moons":
                X, y = make_moons(n_samples=400, noise=0.2, random_state=42)
                cols = ["feature_0", "feature_1"]
            else:
                iris = load_iris()
                mask = iris.target != 2
                X, y = iris.data[mask], iris.target[mask]
                cols = iris.feature_names
            return X, y, cols
        else:
            if uploaded_file is None:
                return None, None, None
            df = pd.read_csv(uploaded_file)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            if y.dtype == object:
                y = pd.factorize(y)[0]
            cols = list(df.columns[:-1])
            return X, y, cols


    X, y, cols = load_data()

    if X is None:
        st.info("Upload a CSV from the sidebar, or pick a built-in dataset.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Dataset Preview")
            df_prev = pd.DataFrame(X, columns=cols)
            df_prev["target"] = y
            st.dataframe(df_prev.head(10))
            st.write(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")

        if train_btn:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                                random_state=42, stratify=y)

            if model_type == "Perceptron":
                model = Perceptron(n_features=X_train.shape[1], lr=lr, activation=activation_name)
                history = model.fit(X_train, y_train, epochs=epochs)
                test_preds = model.predict(X_test)
            else:
                model = MLP([X_train.shape[1], hidden_neurons, 1], activation=activation_name,
                            lr=lr, task="binary")
                history = model.fit(X_train, y_train.reshape(-1, 1), epochs=epochs)
                test_preds = model.predict(X_test)

            test_acc = np.mean(test_preds == y_test)

            st.subheader("Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Train Accuracy", f"{history['accuracy'][-1]*100:.2f}%")
            c2.metric("Test Accuracy", f"{test_acc*100:.2f}%")
            c3.metric("Final Loss", f"{history['loss'][-1]:.4f}")

            st.subheader("Training Curves")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(history["loss"], color="red")
            ax1.set_title("Loss vs Epochs")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")

            ax2.plot(history["accuracy"], color="green")
            ax2.set_title("Accuracy vs Epochs")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")

            st.pyplot(fig)

            with st.expander("Model config used"):
                st.write({
                    "model": model_type,
                    "activation": activation_name,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "hidden_neurons": hidden_neurons if model_type == "MLP" else None,
                })
        else:
            st.info("click Train Model.")

    st.markdown("---")
    st.caption("Implemented By: Anshika Kumawat")

    with tab_cluster:
        render_clustering()

    with tab_forecast:
        render_forecasting()