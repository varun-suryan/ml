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
DURATION = 2.0  # seconds per sample
NUM_SAMPLES = 250  # per sample (2 seconds * 125 Hz)
TRAIN_SIZE = 1000
VAL_SIZE = 200
TEST_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# ================== DATA GENERATION ==================
def generate_synthetic_waveform(duration=2.0, fs=125, seed=None):
    """
    Generate synthetic sine wave with noise and variable frequency/amplitude
    Returns: signal, peak_labels, trough_labels
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    # Random frequency between 0.5 and 3 Hz (heart rate like)
    freq = np.random.uniform(0.8, 2.5)
    
    # Random amplitude
    amplitude = np.random.uniform(50, 100)
    
    # Random baseline
    baseline = np.random.uniform(40, 80)
    
    # Generate sine wave
    signal = amplitude * np.sin(2 * np.pi * freq * t) + baseline
    
    # Add noise
    noise = np.random.normal(0, 2, n_samples)
    signal = signal + noise
    
    # Detect peaks and troughs
    peaks, _ = find_peaks(signal, distance=int(0.3 * fs))
    troughs, _ = find_peaks(-signal, distance=int(0.3 * fs))
    
    # Create binary labels
    peak_labels = np.zeros(n_samples, dtype=np.float32)
    trough_labels = np.zeros(n_samples, dtype=np.float32)
    
    # Mark peaks and troughs with a window (±3 samples)
    for p in peaks:
        start = max(0, p - 3)
        end = min(n_samples, p + 4)
        peak_labels[start:end] = 1.0
    
    for t_idx in troughs:
        start = max(0, t_idx - 3)
        end = min(n_samples, t_idx + 4)
        trough_labels[start:end] = 1.0
    
    return signal.astype(np.float32), peak_labels, trough_labels


class WaveformDataset(Dataset):
    """Dataset for synthetic waveforms"""
    def __init__(self, size, duration=2.0, fs=125, seed=None):
        self.size = size
        self.duration = duration
        self.fs = fs
        self.seed = seed
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        seed = self.seed + idx if self.seed is not None else None
        signal, peak_labels, trough_labels = generate_synthetic_waveform(
            self.duration, self.fs, seed
        )
        
        # Normalize signal
        signal = (signal - signal.mean()) / (signal.std() + 1e-6)
        
        # Convert to tensors
        signal = torch.from_numpy(signal).unsqueeze(0)  # Add channel dimension
        peak_labels = torch.from_numpy(peak_labels)
        trough_labels = torch.from_numpy(trough_labels)
        
        return signal, peak_labels, trough_labels


# ================== CNN MODEL ==================
class PeakTroughCNN(nn.Module):
    """
    1D CNN for detecting peaks and troughs in time series data
    """
    def __init__(self, input_channels=1):
        super(PeakTroughCNN, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        
        # Output heads for peaks and troughs
        self.peak_head = nn.Conv1d(64, 1, kernel_size=1)
        self.trough_head = nn.Conv1d(64, 1, kernel_size=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Encoder
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.conv4(x)))
        
        # Output heads
        peak_out = torch.sigmoid(self.peak_head(x)).squeeze(1)
        trough_out = torch.sigmoid(self.trough_head(x)).squeeze(1)
        
        return peak_out, trough_out


# ================== TRAINING ==================
def train_model(model, train_loader, val_loader, epochs, device):
    """Train the CNN model"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    
    print("\n" + "="*60)
    print("TRAINING CNN MODEL")
    print("="*60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for signals, peak_labels, trough_labels in train_loader:
            signals = signals.to(device)
            peak_labels = peak_labels.to(device)
            trough_labels = trough_labels.to(device)
            
            optimizer.zero_grad()
            peak_pred, trough_pred = model(signals)
            
            loss = criterion(peak_pred, peak_labels) + criterion(trough_pred, trough_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for signals, peak_labels, trough_labels in val_loader:
                signals = signals.to(device)
                peak_labels = peak_labels.to(device)
                trough_labels = trough_labels.to(device)
                
                peak_pred, trough_pred = model(signals)
                loss = criterion(peak_pred, peak_labels) + criterion(trough_pred, trough_labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses


# ================== EVALUATION ==================
def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_signals = []
    all_peak_labels = []
    all_trough_labels = []
    all_peak_preds = []
    all_trough_preds = []
    
    with torch.no_grad():
        for signals, peak_labels, trough_labels in test_loader:
            signals = signals.to(device)
            peak_pred, trough_pred = model(signals)
            
            all_signals.extend(signals.cpu().numpy())
            all_peak_labels.extend(peak_labels.numpy())
            all_trough_labels.extend(trough_labels.numpy())
            all_peak_preds.extend(peak_pred.cpu().numpy())
            all_trough_preds.extend(trough_pred.cpu().numpy())
    
    # Calculate metrics
    peak_preds_binary = (np.array(all_peak_preds) > 0.5).astype(int)
    trough_preds_binary = (np.array(all_trough_preds) > 0.5).astype(int)
    
    peak_labels_flat = np.array(all_peak_labels).flatten()
    trough_labels_flat = np.array(all_trough_labels).flatten()
    peak_preds_flat = peak_preds_binary.flatten()
    trough_preds_flat = trough_preds_binary.flatten()
    
    # Peak metrics
    peak_tp = np.sum((peak_labels_flat == 1) & (peak_preds_flat == 1))
    peak_fp = np.sum((peak_labels_flat == 0) & (peak_preds_flat == 1))
    peak_fn = np.sum((peak_labels_flat == 1) & (peak_preds_flat == 0))
    
    peak_precision = peak_tp / (peak_tp + peak_fp) if (peak_tp + peak_fp) > 0 else 0
    peak_recall = peak_tp / (peak_tp + peak_fn) if (peak_tp + peak_fn) > 0 else 0
    peak_f1 = 2 * peak_precision * peak_recall / (peak_precision + peak_recall) if (peak_precision + peak_recall) > 0 else 0
    
    # Trough metrics
    trough_tp = np.sum((trough_labels_flat == 1) & (trough_preds_flat == 1))
    trough_fp = np.sum((trough_labels_flat == 0) & (trough_preds_flat == 1))
    trough_fn = np.sum((trough_labels_flat == 1) & (trough_preds_flat == 0))
    
    trough_precision = trough_tp / (trough_tp + trough_fp) if (trough_tp + trough_fp) > 0 else 0
    trough_recall = trough_tp / (trough_tp + trough_fn) if (trough_tp + trough_fn) > 0 else 0
    trough_f1 = 2 * trough_precision * trough_recall / (trough_precision + trough_recall) if (trough_precision + trough_recall) > 0 else 0
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    print(f"\nPeak Detection:")
    print(f"  Precision: {peak_precision:.4f}")
    print(f"  Recall:    {peak_recall:.4f}")
    print(f"  F1-Score:  {peak_f1:.4f}")
    print(f"\nTrough Detection:")
    print(f"  Precision: {trough_precision:.4f}")
    print(f"  Recall:    {trough_recall:.4f}")
    print(f"  F1-Score:  {trough_f1:.4f}")
    
    return all_signals, all_peak_labels, all_trough_labels, all_peak_preds, all_trough_preds


# ================== VISUALIZATION ==================
def plot_results(train_losses, val_losses, test_signals, test_peak_labels, 
                test_trough_labels, test_peak_preds, test_trough_preds):
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
    t = np.arange(len(signal)) / SAMPLE_RATE
    
    axes[1].plot(t, signal, 'b-', label='Signal', alpha=0.6)
    
    # True labels
    peak_true_idx = np.where(test_peak_labels[idx] > 0.5)[0]
    trough_true_idx = np.where(test_trough_labels[idx] > 0.5)[0]
    if len(peak_true_idx) > 0:
        axes[1].scatter(peak_true_idx / SAMPLE_RATE, signal[peak_true_idx], 
                       c='green', marker='o', s=100, label='True Peaks', zorder=5)
    if len(trough_true_idx) > 0:
        axes[1].scatter(trough_true_idx / SAMPLE_RATE, signal[trough_true_idx], 
                       c='red', marker='v', s=100, label='True Troughs', zorder=5)
    
    # Predictions
    peak_pred_idx = np.where(test_peak_preds[idx] > 0.5)[0]
    trough_pred_idx = np.where(test_trough_preds[idx] > 0.5)[0]
    if len(peak_pred_idx) > 0:
        axes[1].scatter(peak_pred_idx / SAMPLE_RATE, signal[peak_pred_idx], 
                       c='lime', marker='x', s=100, label='Pred Peaks', zorder=4)
    if len(trough_pred_idx) > 0:
        axes[1].scatter(trough_pred_idx / SAMPLE_RATE, signal[trough_pred_idx], 
                       c='orange', marker='x', s=100, label='Pred Troughs', zorder=4)
    
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Example Prediction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_peak_trough_detection.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'cnn_peak_trough_detection.png'")
    plt.close()


# ================== MAIN ==================
def main():
    print("\n" + "="*60)
    print("CNN-BASED PEAK AND TROUGH DETECTION")
    print("="*60)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Training samples: {TRAIN_SIZE}")
    print(f"Validation samples: {VAL_SIZE}")
    print(f"Test samples: {TEST_SIZE}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = WaveformDataset(TRAIN_SIZE, DURATION, SAMPLE_RATE, seed=42)
    val_dataset = WaveformDataset(VAL_SIZE, DURATION, SAMPLE_RATE, seed=1000)
    test_dataset = WaveformDataset(TEST_SIZE, DURATION, SAMPLE_RATE, seed=2000)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    model = PeakTroughCNN().to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS, device)
    
    # Evaluate model
    test_signals, test_peak_labels, test_trough_labels, test_peak_preds, test_trough_preds = \
        evaluate_model(model, test_loader, device)
    
    # Plot results
    plot_results(train_losses, val_losses, test_signals, test_peak_labels, 
                test_trough_labels, test_peak_preds, test_trough_preds)
    
    # Save model
    torch.save(model.state_dict(), 'peak_trough_cnn_model.pth')
    print(f"\nModel saved as 'peak_trough_cnn_model.pth'")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()