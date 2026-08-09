"""
Evaluation utilities for downstream classification.
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt


class Evaluator:
    """
    Downstream evaluator.
    """

    def __init__(
        self,
        backbone,
        classifier,
        device,
    ):

        self.backbone = backbone.to(device)
        self.classifier = classifier.to(device)

        self.device = device

        self.backbone.eval()
        self.classifier.eval()

    @torch.no_grad()
    def extract_features(
        self,
        dataloader,
    ):

        features = []
        labels = []

        for images, targets in dataloader:

            images = images.to(self.device)

            embedding = self.backbone(images)

            features.append(
                embedding.cpu()
            )

            labels.append(
                targets
            )

        return (
            torch.cat(features),
            torch.cat(labels),
        )

    @torch.no_grad()
    def predict(
        self,
        dataloader,
    ):

        predictions = []
        labels = []

        for images, targets in dataloader:

            images = images.to(self.device)

            logits = self.classifier(
                self.backbone(images)
            )

            preds = logits.argmax(1)

            predictions.extend(
                preds.cpu().numpy()
            )

            labels.extend(
                targets.numpy()
            )

        return (
            np.array(predictions),
            np.array(labels),
        )

    def evaluate(
        self,
        dataloader,
    ):

        preds, labels = self.predict(
            dataloader
        )

        metrics = {

            "accuracy":
            accuracy_score(labels, preds),

            "precision":
            precision_score(
                labels,
                preds,
                average="macro",
            ),

            "recall":
            recall_score(
                labels,
                preds,
                average="macro",
            ),

            "f1":
            f1_score(
                labels,
                preds,
                average="macro",
            ),
        }

        return metrics

    def classification_report(
        self,
        dataloader,
    ):

        preds, labels = self.predict(
            dataloader
        )

        return classification_report(
            labels,
            preds,
        )

    def confusion_matrix(
        self,
        dataloader,
    ):

        preds, labels = self.predict(
            dataloader
        )

        return confusion_matrix(
            labels,
            preds,
        )

    def save_metrics(
        self,
        metrics,
        output_dir,
    ):

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_dir / "metrics.json",
            "w",
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4,
            )

    def save_confusion_matrix(
        self,
        cm,
        output_dir,
    ):

        plt.figure(figsize=(8,8))

        plt.imshow(cm)

        plt.colorbar()

        plt.xlabel("Predicted")

        plt.ylabel("True")

        plt.tight_layout()

        plt.savefig(
            Path(output_dir)
            / "confusion_matrix.png"
        )

        plt.close()

    def plot_tsne(
        self,
        dataloader,
        output_dir,
    ):

        features, labels = self.extract_features(
            dataloader
        )

        embedding = TSNE(
            n_components=2,
            random_state=42,
        ).fit_transform(
            features.numpy()
        )

        plt.figure(figsize=(8,8))

        plt.scatter(
            embedding[:,0],
            embedding[:,1],
            c=labels,
            s=5,
        )

        plt.savefig(
            Path(output_dir)
            / "tsne.png"
        )

        plt.close()