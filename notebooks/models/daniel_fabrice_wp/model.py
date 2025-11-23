"""
LSTM Model Definition
Based on: "Financial Time Series Data Processing for Machine Learning" by Fabrice Daniel (2019)

This module contains the LSTM classifier model.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import mlflow


class SimpleLSTMClassifier(pl.LightningModule):
    """
    Simple LSTM classifier based on paper Section 3, Table 2.
    Paper uses: "1 LSTM 64 units (tanh), 1 Dense output 2 units (softmax)"

    We extend to 3 classes for QClass (Up/Neutral/Down).
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_classes: int = 3, learning_rate: float = 0.001,
                 num_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        self.fc = nn.Linear(hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Store predictions for test set
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        # Classification
        logits = self.fc(last_output)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Store predictions and targets for confusion matrix
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return {'test_loss': loss, 'test_acc': acc}

    def on_test_epoch_end(self):
        # Calculate and log per-class metrics
        target_names = ['Up', 'Neutral', 'Down']
        report = classification_report(self.test_targets, self.test_predictions,
                                       target_names=target_names, output_dict=True)

        # Log per-class metrics to MLflow
        for class_name in target_names:
            mlflow.log_metric(f"test_{class_name.lower()}_precision", report[class_name]['precision'])
            mlflow.log_metric(f"test_{class_name.lower()}_recall", report[class_name]['recall'])
            mlflow.log_metric(f"test_{class_name.lower()}_f1", report[class_name]['f1-score'])

        # Log overall metrics
        mlflow.log_metric("test_accuracy", report['accuracy'])
        mlflow.log_metric("test_macro_avg_f1", report['macro avg']['f1-score'])
        mlflow.log_metric("test_weighted_avg_f1", report['weighted avg']['f1-score'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            # 'monitor': 'val_loss'
            "monitor": 'val_acc'
        }
