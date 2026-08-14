"""
Embedding visualizations.

Generates:

- t-SNE
- UMAP

from saved feature embeddings.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
import umap


class EmbeddingVisualizer:

    def __init__(
        self,
        model_name: str,
        dataset_name,
        output_dir: str = "results",
    ):

        self.model_name = model_name.lower()

        self.output_dir = (Path(output_dir)/ dataset_name.lower())

        self.embedding_dir = (
            self.output_dir /
            "embeddings"
        )

        self.tsne_dir = (
            self.output_dir /
            "tsne"
        )

        self.umap_dir = (
            self.output_dir /
            "umap"
        )

        self.tsne_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.umap_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.features = np.load(
            self.embedding_dir /
            f"{self.model_name}_features.npy"
        )

        self.labels = np.load(
            self.embedding_dir /
            f"{self.model_name}_labels.npy"
        )

        print("\n========== EMBEDDING CHECK ==========")
        print("Shape:", self.features.shape)
        print("Dtype:", self.features.dtype)
        print("NaN :", np.isnan(self.features).any())
        print("Inf :", np.isinf(self.features).any())
        print("Min :", self.features.min())
        print("Max :", self.features.max())
        print("Mean:", self.features.mean())
        print("Std :", self.features.std())
        print("=====================================\n")

        if np.isnan(self.features).any():
            raise ValueError("NaN detected in embeddings.")

        if np.isinf(self.features).any():
            raise ValueError("Inf detected in embeddings.")

        from sklearn.decomposition import PCA

        n_components = min(
            50,
            self.features.shape[1],
            self.features.shape[0] - 1,
        )

        self.features = PCA(
            n_components=n_components,
            random_state=42,
        ).fit_transform(self.features)

    def generate_tsne(self):

        print("Generating t-SNE...")

        tsne = TSNE(

            n_components=2,

            perplexity=30,

            learning_rate="auto",

            init="pca",

            random_state=42,

        )

        embedding = tsne.fit_transform(
            self.features
        )

        plt.figure(
            figsize=(8,8)
        )

        scatter = plt.scatter(

            embedding[:,0],

            embedding[:,1],

            c=self.labels,

            s=8,

            cmap="tab10",

        )

        plt.legend(
            *scatter.legend_elements(),
            title="Class",
            loc="best",
        )

        plt.title(
            f"{self.model_name.upper()} t-SNE"
        )

        plt.tight_layout()

        plt.savefig(

            self.tsne_dir /
            f"{self.model_name}_tsne.png",

            dpi=300,

        )

        plt.close()

    def generate_umap(self):

        print("Generating UMAP...")

        reducer = umap.UMAP(

            n_neighbors=15,

            min_dist=0.1,

            metric="euclidean",

            random_state=42,

        )

        embedding = reducer.fit_transform(
            self.features
        )

        plt.figure(
            figsize=(8,8)
        )

        scatter = plt.scatter(

            embedding[:,0],

            embedding[:,1],

            c=self.labels,

            s=8,

            cmap="tab10",

        )

        plt.legend(
            *scatter.legend_elements(),
            title="Class",
            loc="best",
        )

        plt.title(
            f"{self.model_name.upper()} UMAP"
        )

        plt.tight_layout()

        plt.savefig(

            self.umap_dir /
            f"{self.model_name}_umap.png",

            dpi=300,

        )

        plt.close()

    def generate_all(self):

        self.generate_tsne()

        self.generate_umap()

        print(
            "Embedding visualizations saved."
        )