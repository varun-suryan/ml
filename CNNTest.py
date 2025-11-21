import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks

# ================== CONFIGURATION ==================
SAMPLE_RATE = 125  # Hz
DURATION = 2.0  # seconds per window
NUM_SAMPLES = 250  # per window (2 seconds * 125 Hz)
LONG_SIGNAL_DURATION = 300.0  # Generate one 300 second signal (5 minutes)
WINDOW_STRIDE = 25  # Stride for sliding window (25 samples = 0.2 seconds)
# Split the signal: first 240s for train, next 30s for val, last 30s for test
TRAIN_END_TIME = 240.0
VAL_END_TIME = 270.0
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005

# ================== DATA GENERATION ==================
def generate_long_synthetic_waveform(duration=60.0, fs=125, seed=None):
    """
    Generate a long synthetic sine wave with varying frequency and amplitude
    Returns: signal, combined_labels (0=background, 1=peak, 2=trough)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    # Generate signal with varying frequency and amplitude
    signal = np.zeros(n_samples)
    
    # Divide into segments with different characteristics
    n_segments = 5
    segment_length = n_samples // n_segments
    
    for i in range(n_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length if i < n_segments - 1 else n_samples
        segment_t = t[start_idx:end_idx]
        
        # Random frequency between 0.8 and 2.5 Hz (heart rate like)
        freq = np.random.uniform(0.8, 2.5)
        
        # Random amplitude
        amplitude = np.random.uniform(50, 100)
        
        # Random baseline
        baseline = np.random.uniform(40, 80)
        
        # Generate sine wave for this segment
        segment_signal = amplitude * np.sin(2 * np.pi * freq * segment_t) + baseline
        signal[start_idx:end_idx] = segment_signal
    
    # Add noise
    noise = np.random.normal(0, 2, n_samples)
    signal = signal + noise
    
    # Detect peaks and troughs
    peaks, _ = find_peaks(signal, distance=int(0.3 * fs))
    troughs, _ = find_peaks(-signal, distance=int(0.3 * fs))
    
    # Create binary labels: 0=background, 1=landmark (peak or trough)
    combined_labels = np.zeros(n_samples, dtype=np.int64)
    
    # Mark peaks and troughs with a window (±3 samples)
    for p in peaks:
        start = max(0, p - 3)
        end = min(n_samples, p + 4)
        combined_labels[start:end] = 1  # Landmark class
    
    for t_idx in troughs:
        start = max(0, t_idx - 3)
        end = min(n_samples, t_idx + 4)
        combined_labels[start:end] = 1  # Landmark class
    
    return signal.astype(np.float32), combined_labels


class SlidingWindowDataset(Dataset):
    """Dataset using sliding windows over a single long signal"""
    def __init__(self, signal, combined_labels, start_time, end_time,
                 window_duration=2.0, stride=50, fs=125):
        self.signal = signal
        self.combined_labels = combined_labels
        self.window_duration = window_duration
        self.stride = stride
        self.fs = fs
        
        self.window_size = int(window_duration * fs)
        
        # Calculate start and end indices for this split
        self.start_idx = int(start_time * fs)
        self.end_idx = int(end_time * fs)
        
        # Calculate number of windows in this range
        available_length = self.end_idx - self.start_idx
        self.num_windows = (available_length - self.window_size) // stride + 1
        
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        # Calculate window start position
        window_start = self.start_idx + idx * self.stride
        window_end = window_start + self.window_size
        
        # Extract window
        signal_window = self.signal[window_start:window_end]
        label_window = self.combined_labels[window_start:window_end]
        
        # Normalize signal window
        signal_window = (signal_window - signal_window.mean()) / (signal_window.std() + 1e-6)
        
        # Convert to tensors
        signal_window = torch.from_numpy(signal_window).unsqueeze(0)  # Add channel dimension
        label_window = torch.from_numpy(label_window)
        
        return signal_window, label_window


# ================== MODELS ==================

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


class TransformerPeakTroughDetector(nn.Module):
    """
    Transformer-based model for landmark detection
    Outputs: 2-class prediction (background, landmark)
    """
    def __init__(self, input_channels=1, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, num_classes=2):
        super(TransformerPeakTroughDetector, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection for 3-class classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        
        # Project input
        x = self.input_proj(x)  # (batch, d_model, seq_len)
        
        # Transpose for transformer
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Classify each timestep
        logits = self.classifier(x)  # (batch, seq_len, num_classes)
        
        return logits


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


# ================== TRAINING ==================
def train_model(model, train_loader, val_loader, epochs, device):
    """Train the model with binary classification"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(signals)
            
            # Reshape for loss computation
            logits = logits.reshape(-1, 2)  # (batch*seq_len, 2)
            labels = labels.reshape(-1)  # (batch*seq_len,)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                
                logits = model(signals)
                logits = logits.reshape(-1, 2)
                labels = labels.reshape(-1)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses


# ================== EVALUATION ==================
def evaluate_model(model, test_loader, device):
    """Evaluate model on test set with binary classification"""
    model.eval()
    
    all_signals = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            logits = model(signals)
            preds = torch.argmax(logits, dim=-1)
            
            all_signals.extend(signals.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Flatten for metrics
    labels_flat = np.array(all_labels).flatten()
    preds_flat = np.array(all_preds).flatten()
    
    # Calculate metrics
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    # Landmark detection metrics (class 1)
    tp = np.sum((labels_flat == 1) & (preds_flat == 1))
    fp = np.sum((labels_flat == 0) & (preds_flat == 1))
    fn = np.sum((labels_flat == 1) & (preds_flat == 0))
    tn = np.sum((labels_flat == 0) & (preds_flat == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(labels_flat)
    
    print(f"\nLandmark Detection (Peaks & Troughs):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Negatives: {fn}")
    
    return all_signals, all_labels, all_preds


# ================== VISUALIZATION ==================
def plot_results(train_losses, val_losses, test_signals, test_labels, test_preds):
    """Plot training curves and example predictions"""
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot example predictions
    idx = 0
    signal = test_signals[idx][0]  # Remove channel dimension
    labels = test_labels[idx]
    preds = test_preds[idx]
    t = np.arange(len(signal)) / SAMPLE_RATE
    
    axes[1].plot(t, signal, 'b-', label='Signal', alpha=0.6, linewidth=1.5)
    
    # True labels (landmarks = peaks or troughs)
    landmark_true_idx = np.where(labels == 1)[0]
    if len(landmark_true_idx) > 0:
        axes[1].scatter(landmark_true_idx / SAMPLE_RATE, signal[landmark_true_idx], 
                       c='green', marker='o', s=100, label='True Landmarks', zorder=5, edgecolors='darkgreen', linewidth=2)
    
    # Predictions
    landmark_pred_idx = np.where(preds == 1)[0]
    if len(landmark_pred_idx) > 0:
        axes[1].scatter(landmark_pred_idx / SAMPLE_RATE, signal[landmark_pred_idx], 
                       c='red', marker='x', s=150, label='Pred Landmarks', zorder=4, linewidth=3)
    
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Example Prediction (Binary Classification: Background vs Landmarks)')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_peak_trough_detection.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'cnn_peak_trough_detection.png'")
    plt.close()


# ================== MAIN ==================
def main():
    print("\n" + "="*60)
    print("ADVANCED PEAK AND TROUGH DETECTION")
    print("="*60)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Window Duration: {DURATION} seconds")
    print(f"Long Signal Duration: {LONG_SIGNAL_DURATION} seconds")
    print(f"Window Stride: {WINDOW_STRIDE} samples ({WINDOW_STRIDE/SAMPLE_RATE:.2f} seconds)")
    print(f"\nData Split:")
    print(f"  Training:   0.0 - {TRAIN_END_TIME} seconds")
    print(f"  Validation: {TRAIN_END_TIME} - {VAL_END_TIME} seconds")
    print(f"  Test:       {VAL_END_TIME} - {LONG_SIGNAL_DURATION} seconds")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate single long signal
    print("\nGenerating single 300-second waveform...")
    signal, combined_labels = generate_long_synthetic_waveform(
        LONG_SIGNAL_DURATION, SAMPLE_RATE, seed=42
    )
    print(f"Signal generated: {len(signal)} samples")
    print(f"Total landmarks detected: {np.sum(combined_labels == 1)}")
    print(f"Background samples: {np.sum(combined_labels == 0)}")
    
    # Create datasets with sliding windows over different time ranges
    train_dataset = SlidingWindowDataset(
        signal, combined_labels,
        start_time=0.0, end_time=TRAIN_END_TIME,
        window_duration=DURATION, stride=WINDOW_STRIDE, fs=SAMPLE_RATE
    )
    val_dataset = SlidingWindowDataset(
        signal, combined_labels,
        start_time=TRAIN_END_TIME, end_time=VAL_END_TIME,
        window_duration=DURATION, stride=WINDOW_STRIDE, fs=SAMPLE_RATE
    )
    test_dataset = SlidingWindowDataset(
        signal, combined_labels,
        start_time=VAL_END_TIME, end_time=LONG_SIGNAL_DURATION,
        window_duration=DURATION, stride=WINDOW_STRIDE, fs=SAMPLE_RATE
    )
    
    print(f"\nTotal training windows: {len(train_dataset)}")
    print(f"Total validation windows: {len(val_dataset)}")
    print(f"Total test windows: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Try different architectures
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE: Hybrid CNN-Transformer")
    print("="*60)
    
    # Create model - using Hybrid CNN-Transformer (best of both worlds)
    model = HybridCNNTransformer(
        input_channels=1,
        d_model=128,
        nhead=8,
        num_layers=3,
        num_classes=2
    ).to(device)
    
    # Uncomment to try pure Transformer:
    # model = TransformerPeakTroughDetector(
    #     input_channels=1,
    #     d_model=128,
    #     nhead=8,
    #     num_layers=4,
    #     dim_feedforward=512,
    #     num_classes=2
    # ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS, device)
    
    # Evaluate model
    test_signals, test_labels, test_preds = evaluate_model(model, test_loader, device)
    
    # Plot results
    plot_results(train_losses, val_losses, test_signals, test_labels, test_preds)
    
    # Save model
    torch.save(model.state_dict(), 'peak_trough_cnn_model.pth')
    print(f"\nModel saved as 'peak_trough_cnn_model.pth'")
    print("\n" + "="*60)
    print("\nIMPROVEMENTS IMPLEMENTED:")
    print("1. Single 300-second waveform with temporal continuity")
    print("2. Sliding window approach (2s windows, 0.2s stride)")
    print("3. Temporal train/val/test split (240s/30s/30s)")
    print("4. Binary classification (background vs landmarks)")
    print("5. Hybrid CNN-Transformer architecture")
    print("6. Positional encoding for temporal awareness")
    print("7. Multi-head attention for global context")
    print("="*60)


if __name__ == "__main__":
    main()