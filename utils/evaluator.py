"""
Evaluation utilities for Linear Probing.

Computes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Inference Time
- Parameter Count

Also saves:

- Confusion Matrix (.csv/.png)
- Embeddings (.npy)
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


class Evaluator:

    def __init__(
        self,
        backbone,
        classifier,
        dataloader,
        device,
        model_name: str,
        dataset_name,
        output_dir: str = "results",
    ):

        self.backbone = backbone
        self.classifier = classifier
        self.dataloader = dataloader
        self.device = device

        self.model_name = model_name.lower()

        self.output_dir = (Path(output_dir)/ dataset_name.lower())

        self.cm_dir = (
            self.output_dir /
            "confusion_matrix"
        )

        self.embedding_dir = (
            self.output_dir /
            "embeddings"
        )

        self.cm_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embedding_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


    @torch.no_grad()
    def evaluate(self):

        self.backbone.eval()
        self.classifier.eval()

        predictions = []
        labels = []

        features = []

        start = time.perf_counter()

        for images, target in self.dataloader:

            images = images.to(self.device)
            target = target.to(self.device)

            embedding = self.backbone(images)

            logits = self.classifier(embedding)

            pred = logits.argmax(dim=1)

            predictions.extend(
                pred.cpu().numpy()
            )

            labels.extend(
                target.cpu().numpy()
            )

            features.append(
                embedding.cpu().numpy()
            )

        inference_time = (
            time.perf_counter() - start
        )

        features = np.concatenate(
            features,
            axis=0,
        )

        labels = np.array(labels)
        predictions = np.array(predictions)

        # ==========================================================
        # Metrics
        # ==========================================================

        accuracy = accuracy_score(
            labels,
            predictions,
        )

        precision = precision_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        # ==========================================================
        # Confusion Matrix
        # ==========================================================

        cm = confusion_matrix(
            labels,
            predictions,
        )

        np.savetxt(
            self.cm_dir /
            f"{self.model_name}_cm.csv",
            cm,
            delimiter=",",
            fmt="%d",
        )

        fig, ax = plt.subplots(
            figsize=(8, 8),
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
        )

        disp.plot(
            ax=ax,
            colorbar=False,
        )

        plt.title(
            f"{self.model_name.upper()} Confusion Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            self.cm_dir /
            f"{self.model_name}_cm.png",
            dpi=300,
        )

        plt.close(fig)

        # ==========================================================
        # Save Embeddings
        # ==========================================================

        np.save(
            self.embedding_dir /
            f"{self.model_name}_features.npy",
            features,
        )

        np.save(
            self.embedding_dir /
            f"{self.model_name}_labels.npy",
            labels,
        )

        # ==========================================================
        # Model Statistics
        # ==========================================================

        backbone_parameters = sum(
            p.numel()
            for p in self.backbone.parameters()
        )

        classifier_parameters = sum(
            p.numel()
            for p in self.classifier.parameters()
        )

        total_parameters = (
            backbone_parameters +
            classifier_parameters
        )

        # ==========================================================
        # Build Metrics Dictionary
        # ==========================================================

        metrics = {

            "accuracy": float(accuracy),

            "precision": float(precision),

            "recall": float(recall),

            "f1_score": float(f1),

            "confusion_matrix": cm,

            "inference_time": float(inference_time),

            "backbone_parameters": int(backbone_parameters),

            "classifier_parameters": int(classifier_parameters),

            "total_parameters": int(total_parameters),

            "features": features,

            "labels": labels,
        }

        # ==========================================================
        # Console Summary
        # ==========================================================

        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)

        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"F1 Score       : {f1:.4f}")
        print(f"Inference Time : {inference_time:.4f} sec")
        print(f"Backbone Parameters  : {backbone_parameters:,}")
        print(f"Classifier Parameters: {classifier_parameters:,}")
        print(f"Total Parameters     : {total_parameters:,}")

        print("=" * 60)

        return metrics