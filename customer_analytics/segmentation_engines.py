import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

def run_kmeans(X, n_clusters=4, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return labels, model

def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model

def run_agglomerative(X, n_clusters=4, linkage="ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return labels, model

def cluster_score(X, labels):
    unique_labels = set(labels)
    unique_labels.discard(-1)
    if len(unique_labels) < 2:
        return None
    mask = labels != -1
    if mask.sum() < 2:
        return None
    return silhouette_score(X[mask], labels[mask])

def reduce_dimensions(X, method="PCA", n_components=2, random_state=42):
    if method == "PCA":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == "t-SNE":
        perplexity = min(30, max(5, X.shape[0] // 4))
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, init="pca")
    elif method == "UMAP":
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is not installed. Run: pip install umap-learn")
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")
    return reducer.fit_transform(X)

def segment_customers(df, feature_cols, algorithm="K-Means", n_clusters=4, eps=0.5, min_samples=5, linkage="ward"):
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if algorithm == "K-Means":
        labels, model = run_kmeans(X_scaled, n_clusters=n_clusters)
    elif algorithm == "DBSCAN":
        labels, model = run_dbscan(X_scaled, eps=eps, min_samples=min_samples)
    elif algorithm == "Agglomerative":
        labels, model = run_agglomerative(X_scaled, n_clusters=n_clusters, linkage=linkage)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    score = cluster_score(X_scaled, labels)
    return labels, model, X_scaled, score

def render():
    import streamlit as st
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    st.subheader("Customer Segmentation (Clustering)")
    st.caption("K-Means, DBSCAN & Agglomerative Clustering with PCA / t-SNE / UMAP visualization")
    data_source = st.radio("Dataset Source", ["Built-in Demo Data", "Upload CSV"], key="cluster_source")
    if data_source == "Built-in Demo Data":
        X_demo, _ = make_blobs(n_samples=400, centers=4, n_features=4, cluster_std=1.3, random_state=42)
        df = pd.DataFrame(X_demo, columns=[f"feature_{i}" for i in range(X_demo.shape[1])])
    else:
        uploaded = st.file_uploader("Upload customer CSV (numeric columns only)", type=["csv"], key="cluster_upload")
        if uploaded is None:
            st.info("Upload a CSV to continue, or switch to the built-in demo data.")
            return
        df = pd.read_csv(uploaded)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = st.multiselect("Features to use for clustering", numeric_cols, default=numeric_cols)
    if len(feature_cols) < 2:
        st.warning("Select at least 2 numeric features.")
        return
    algorithm = st.selectbox("Clustering Algorithm", ["K-Means", "DBSCAN", "Agglomerative"])
    n_clusters, eps, min_samples, linkage = 4, 0.5, 5, "ward"
    if algorithm in ("K-Means", "Agglomerative"):
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        if algorithm == "Agglomerative":
            linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
    else:
        eps = st.slider("DBSCAN eps", 0.1, 3.0, 0.5, step=0.1)
        min_samples = st.slider("DBSCAN min_samples", 2, 20, 5)
    reduction_method = st.selectbox("Dimensionality Reduction (for plotting)", ["PCA", "t-SNE", "UMAP"])
    if st.button("Run Clustering"):
        labels, model, X_scaled, score = segment_customers(df, feature_cols, algorithm=algorithm, n_clusters=n_clusters, eps=eps, min_samples=min_samples, linkage=linkage)
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters Found", n_found)
        c2.metric("Silhouette Score", f"{score:.3f}" if score is not None else "N/A")
        c3.metric("Noise Points", int((labels == -1).sum()) if algorithm == "DBSCAN" else 0)
        try:
            coords = reduce_dimensions(X_scaled, method=reduction_method)
            fig, ax = plt.subplots(figsize=(7, 5))
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=25, alpha=0.8)
            ax.set_title(f"Customer Segments — {reduction_method} projection")
            ax.set_xlabel(f"{reduction_method} 1")
            ax.set_ylabel(f"{reduction_method} 2")
            legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
            ax.add_artist(legend)
            st.pyplot(fig)
        except ImportError as e:
            st.error(str(e))
        result_df = df.copy()
        result_df["cluster"] = labels
        with st.expander("Segmented data"):
            st.dataframe(result_df.head(20))
        st.download_button("Download segmented CSV", result_df.to_csv(index=False), file_name="customer_segments.csv")