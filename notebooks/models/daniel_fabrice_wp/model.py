"""
LSTM Model Definition
Based on: "Financial Time Series Data Processing for Machine Learning" by Fabrice Daniel (2019)

This module contains LSTM classifier models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            optimizer, mode='max', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_acc'
        }


class AdvancedLSTMWithAttention(pl.LightningModule):
    """
    Advanced LSTM classifier with multiple intelligent features:
    - Bidirectional LSTM layers for capturing both past and future context
    - Multi-head self-attention mechanism for capturing important temporal dependencies
    - Residual connections to help with gradient flow
    - Layer normalization for stable training
    - Adaptive feature extraction with 1D convolutions
    - Advanced regularization (dropout, layer dropout, weight decay)
    - Gradient clipping to prevent exploding gradients
    
    This architecture is inspired by modern time series forecasting approaches
    and combines elements from Transformer and LSTM architectures.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_classes: int = 3,
        num_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.3,
        layer_dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        use_conv: bool = True,
        conv_kernel_size: int = 3
    ):
        """
        Initialize the Advanced LSTM with Attention model.
        
        Args:
            n_features: Number of input features
            hidden_size: Hidden size for LSTM layers (must be divisible by num_attention_heads)
            num_classes: Number of output classes (default 3 for Up/Neutral/Down)
            num_layers: Number of stacked LSTM layers
            num_attention_heads: Number of attention heads
            dropout: Dropout probability for LSTM and linear layers
            layer_dropout: Probability of dropping entire layers during training
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            use_conv: Whether to use 1D convolution for feature extraction
            conv_kernel_size: Kernel size for convolutional layer
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Validate that hidden_size is divisible by num_attention_heads
        assert hidden_size % num_attention_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
        
        # Optional 1D Convolution for adaptive feature extraction
        self.use_conv = use_conv
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=n_features,
                out_channels=n_features,
                kernel_size=conv_kernel_size,
                padding='same'
            )
            self.conv_norm = nn.LayerNorm(n_features)
        
        # Input projection to ensure proper dimensionality
        self.input_projection = nn.Linear(n_features, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # Divided by 2 because bidirectional doubles the output
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization after attention
        self.attn_norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward network with residual connection
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),  # GELU activation (smoother than ReLU)
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.ff_norm = nn.LayerNorm(hidden_size)
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Loss function with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Layer dropout for stochastic depth regularization
        self.layer_dropout = layer_dropout
        
        # Store predictions for test set
        self.test_predictions = []
        self.test_targets = []
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            logits: Output tensor of shape (batch, num_classes)
        """
        batch_size, seq_len, n_features = x.shape
        
        # Optional 1D convolution for feature extraction
        if self.use_conv:
            # Conv1D expects (batch, channels, seq_len)
            x_conv = x.transpose(1, 2)
            x_conv = self.conv1d(x_conv)
            x_conv = x_conv.transpose(1, 2)
            x = x + self.conv_norm(x_conv)  # Residual connection
        
        # Project input features
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # Bidirectional LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Self-attention with residual connection
        # Apply stochastic layer dropout during training
        if self.training and torch.rand(1).item() < self.layer_dropout:
            attn_out = lstm_out  # Skip attention layer
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            attn_out = lstm_out + attn_out  # Residual connection
        
        attn_out = self.attn_norm(attn_out)
        
        # Feed-forward network with residual connection
        if self.training and torch.rand(1).item() < self.layer_dropout:
            ff_out = attn_out  # Skip feed-forward layer
        else:
            ff_out = self.feed_forward(attn_out)
            ff_out = attn_out + ff_out  # Residual connection
        
        ff_out = self.ff_norm(ff_out)
        
        # Global pooling: Take mean and max of sequence, then concatenate
        mean_pool = torch.mean(ff_out, dim=1)
        max_pool, _ = torch.max(ff_out, dim=1)
        
        # Combine pooled representations
        combined = mean_pool + max_pool  # Element-wise addition
        
        # Classification
        logits = self.classifier(combined)
        
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
        report = classification_report(
            self.test_targets,
            self.test_predictions,
            target_names=target_names,
            output_dict=True
        )
        
        # Log per-class metrics to MLflow
        for class_name in target_names:
            mlflow.log_metric(f"test_{class_name.lower()}_precision", 
                            report[class_name]['precision'])
            mlflow.log_metric(f"test_{class_name.lower()}_recall", 
                            report[class_name]['recall'])
            mlflow.log_metric(f"test_{class_name.lower()}_f1", 
                            report[class_name]['f1-score'])
        
        # Log overall metrics
        mlflow.log_metric("test_accuracy", report['accuracy'])
        mlflow.log_metric("test_macro_avg_f1", report['macro avg']['f1-score'])
        mlflow.log_metric("test_weighted_avg_f1", report['weighted avg']['f1-score'])
    
    def configure_optimizers(self):
        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Number of epochs for first restart
            T_mult=2,  # Multiplication factor for restart period
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
