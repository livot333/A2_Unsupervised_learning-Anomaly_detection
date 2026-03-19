import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Definition of the internal network architecture
class LSTM_AE_Arch(nn.Module):
    def __init__(self, seq_len, hidden_dim, n_features=1):
        super(LSTM_AE_Arch, self).__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        """
        Forward pass: Encoder compresses input, Decoder attempts reconstruction.
        """
        # Encode: Extract the last hidden state as the context vector
        _, (hidden, _) = self.encoder(x)
        
        # Repeat the context vector to match the sequence length
        context = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        
        # Decode: Reconstruct the sequence from the context
        x_decoded, _ = self.decoder(context)
        
        # Map back to original feature dimension (1)
        return self.output_layer(x_decoded)

class LSTM_AE_Detector:
    def __init__(self, seq_len=50, hidden_dim=32, epochs=5, percentile=99, n_features=25):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.percentile = percentile
        self.n_features = n_features
        self.model = LSTM_AE_Arch(seq_len, hidden_dim, n_features).to(self.device)

    def _create_sequences(self, data):
        # Use first column only — primary telemetry signal
        source = data[:, 0].reshape(-1, 1)
        if len(source) < self.seq_len:
            return np.empty((0, self.seq_len, 1))

        # Create sliding windows: (n_windows, seq_len, 1)
        shape   = (source.shape[0] - self.seq_len + 1, self.seq_len, 1)
        strides = (source.strides[0], source.strides[0], source.strides[1])
        return np.lib.stride_tricks.as_strided(source, shape=shape, strides=strides)

    def fit(self, train_data_dict):
        """
        Trains the model on the first column of all provided channel files.
        """
        all_sequences = []
        for data in train_data_dict.values():
            seqs = self._create_sequences(data)
            if seqs.size > 0:
                all_sequences.append(seqs)
        
        # Stack all sequences into a single training tensor
        X_train = np.vstack(all_sequences)
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                optimizer.zero_grad()
                output = self.model(batch[0])
                loss = criterion(output, batch[0])
                loss.backward()
                optimizer.step()
        

    def prediction(self, test_data_dict):
        """
        Calculates reconstruction error and returns a dictionary of outlier indices for each channel.
        """
        self.model.eval()
        outliers_dict = {}

        for cid, data in test_data_dict.items():
            X_test = self._create_sequences(data)
            if X_test.size == 0:
                outliers_dict[cid] = set()
                continue

            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # Reconstruct the sequence
                reconstructed = self.model(X_tensor)
                
                # Calculate Mean Squared Error (MSE) per window
                
                mse = torch.mean((X_tensor - reconstructed)**2, dim=(1, 2)).cpu().numpy()
            
            # Map window scores back to original data length (pad beginning with zeros)
            full_scores = np.zeros(len(data))
            full_scores[self.seq_len - 1:] = mse
            
            # Establish a dynamic threshold based on the specified percentile
            threshold = np.percentile(full_scores, self.percentile)
            predicted_indices = np.where(full_scores > threshold)[0]
            
            # Store predicted anomaly indices as a set
            outliers_dict[cid] = set(predicted_indices)
           

        return outliers_dict