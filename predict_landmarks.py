import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import find_peaks

# ================== CONFIGURATION ==================
SAMPLE_RATE = 125  # Hz
DURATION = 2.0  # seconds per window
NUM_SAMPLES = 250  # per window (2 seconds * 125 Hz)
MODEL_PATH = 'peak_trough_cnn_model.pth'

# ================== MODEL DEFINITION ==================
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture for binary classification (background vs landmark)
    """
    def __init__(self, input_channels=1, d_model=128, nhead=8, num_layers=3, num_classes=2):
        super(HybridCNNTransformer, self).__init__()
        
        # CNN for local feature extraction
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, d_model, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(d_model)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer for global context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(d_model, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # CNN feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Prepare for transformer
        x_t = x.transpose(1, 2)  # (batch, seq_len, d_model)
        x_t = self.pos_encoder(x_t)
        
        # Transformer processing
        x_t = self.transformer(x_t)
        
        # Back to CNN format
        x = x_t.transpose(1, 2)  # (batch, d_model, seq_len)
        
        # Classify
        logits = self.classifier(x)  # (batch, num_classes, seq_len)
        logits = logits.transpose(1, 2)  # (batch, seq_len, num_classes)
        
        return logits


# ================== UTILITY FUNCTIONS ==================
def generate_test_waveform(duration=2.0, fs=125, seed=None):
    """
    Generate a synthetic test waveform with known peaks and troughs
    Returns: signal, true_labels
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    # Random frequency between 0.8 and 2.5 Hz
    freq = np.random.uniform(1.0, 2.0)
    
    # Random amplitude
    amplitude = np.random.uniform(60, 90)
    
    # Random baseline
    baseline = np.random.uniform(50, 70)
    
    # Generate sine wave
    signal = amplitude * np.sin(2 * np.pi * freq * t) + baseline
    
    # Add noise
    noise = np.random.normal(0, 2, n_samples)
    signal = signal + noise
    
    # Detect peaks and troughs
    peaks, _ = find_peaks(signal, distance=int(0.3 * fs))
    troughs, _ = find_peaks(-signal, distance=int(0.3 * fs))
    
    # Create binary labels: 0=background, 1=landmark (peak or trough)
    true_labels = np.zeros(n_samples, dtype=np.int64)
    
    # Mark peaks and troughs with a window (±3 samples)
    for p in peaks:
        start = max(0, p - 3)
        end = min(n_samples, p + 4)
        true_labels[start:end] = 1
    
    for t_idx in troughs:
        start = max(0, t_idx - 3)
        end = min(n_samples, t_idx + 4)
        true_labels[start:end] = 1
    
    return signal.astype(np.float32), true_labels, peaks, troughs


def load_model(model_path, device):
    """Load the trained model"""
    model = HybridCNNTransformer(
        input_channels=1,
        d_model=128,
        nhead=8,
        num_layers=3,
        num_classes=2
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def predict(model, signal, device):
    """
    Make predictions on a signal
    Returns: predicted labels (0 or 1 for each sample)
    """
    # Normalize signal
    signal_normalized = (signal - signal.mean()) / (signal.std() + 1e-6)
    
    # Convert to tensor and add batch and channel dimensions
    signal_tensor = torch.from_numpy(signal_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(signal_tensor)  # (1, seq_len, 2)
        preds = torch.argmax(logits, dim=-1)  # (1, seq_len)
    
    return preds.cpu().numpy().squeeze()


def plot_predictions(signal, true_labels, pred_labels, peaks, troughs, output_path='prediction_results.png'):
    """
    Plot the signal with true and predicted landmarks
    """
    t = np.arange(len(signal)) / SAMPLE_RATE
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot 1: Signal with true landmarks
    axes[0].plot(t, signal, 'b-', label='Signal', alpha=0.7, linewidth=1.5)
    axes[0].scatter(peaks / SAMPLE_RATE, signal[peaks], 
                   c='green', marker='^', s=150, label='True Peaks', zorder=5, edgecolors='darkgreen', linewidth=2)
    axes[0].scatter(troughs / SAMPLE_RATE, signal[troughs], 
                   c='red', marker='v', s=150, label='True Troughs', zorder=5, edgecolors='darkred', linewidth=2)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_title('Ground Truth: Signal with True Peaks and Troughs', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Signal with predicted landmarks
    pred_landmark_idx = np.where(pred_labels == 1)[0]
    axes[1].plot(t, signal, 'b-', label='Signal', alpha=0.7, linewidth=1.5)
    if len(pred_landmark_idx) > 0:
        axes[1].scatter(pred_landmark_idx / SAMPLE_RATE, signal[pred_landmark_idx], 
                       c='orange', marker='x', s=200, label='Predicted Landmarks', zorder=5, linewidth=3)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].set_title('Model Predictions: Detected Landmarks', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Comparison - True vs Predicted labels
    true_landmark_idx = np.where(true_labels == 1)[0]
    axes[2].plot(t, signal, 'b-', label='Signal', alpha=0.5, linewidth=1.5)
    if len(true_landmark_idx) > 0:
        axes[2].scatter(true_landmark_idx / SAMPLE_RATE, signal[true_landmark_idx], 
                       c='green', marker='o', s=100, label='True Landmarks', zorder=5, alpha=0.6)
    if len(pred_landmark_idx) > 0:
        axes[2].scatter(pred_landmark_idx / SAMPLE_RATE, signal[pred_landmark_idx], 
                       c='red', marker='x', s=150, label='Predicted Landmarks', zorder=4, linewidth=2)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Amplitude', fontsize=12)
    axes[2].set_title('Comparison: True vs Predicted Landmarks', fontsize=14, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Prediction plot saved as '{output_path}'")
    plt.close()


def calculate_metrics(true_labels, pred_labels):
    """Calculate and print performance metrics"""
    tp = np.sum((true_labels == 1) & (pred_labels == 1))
    fp = np.sum((true_labels == 0) & (pred_labels == 1))
    tn = np.sum((true_labels == 0) & (pred_labels == 0))
    fn = np.sum((true_labels == 1) & (pred_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(true_labels)
    
    print("\n" + "="*60)
    print("PREDICTION METRICS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Negatives: {fn}")
    print("="*60 + "\n")


# ================== MAIN ==================
def main():
    print("\n" + "="*60)
    print("LANDMARK DETECTION - INFERENCE")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    print(f"\nLoading model from '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH, device)
    
    # Generate test waveform
    print(f"\nGenerating {DURATION}-second test waveform...")
    signal, true_labels, peaks, troughs = generate_test_waveform(DURATION, SAMPLE_RATE, seed=123)
    print(f"Test signal generated: {len(signal)} samples")
    print(f"True peaks: {len(peaks)}")
    print(f"True troughs: {len(troughs)}")
    print(f"Total landmarks: {np.sum(true_labels == 1)} samples")
    
    # Make predictions
    print("\nMaking predictions...")
    pred_labels = predict(model, signal, device)
    print(f"Predicted landmarks: {np.sum(pred_labels == 1)} samples")
    
    # Calculate metrics
    calculate_metrics(true_labels, pred_labels)
    
    # Plot results
    print("Generating visualization...")
    plot_predictions(signal, true_labels, pred_labels, peaks, troughs)
    
    print("Inference complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
