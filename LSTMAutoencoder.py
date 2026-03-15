import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Definice vnitřní architektury sítě
class LSTM_AE_Arch(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super(LSTM_AE_Arch, self).__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(1, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        context = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        x_decoded, _ = self.decoder(context)
        return self.output_layer(x_decoded)

class LSTM_AE_Detector:
    def __init__(self, seq_len=50, hidden_dim=32, epochs=5, percentile=99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.percentile = percentile
        self.model = LSTM_AE_Arch(seq_len, hidden_dim).to(self.device)

    def _create_sequences(self, data):
        # Pouze první sloupec
        source = data[:, 0].reshape(-1, 1)
        if len(source) < self.seq_len:
            return np.empty((0, self.seq_len, 1))
        
        # Vytvoření oken (stride 1)
        shape = (source.shape[0] - self.seq_len + 1, self.seq_len, 1)
        strides = (source.strides[0], source.strides[0], source.strides[1])
        return np.lib.stride_tricks.as_strided(source, shape=shape, strides=strides)

    def fit(self, train_data_dict):
        """Trénink na prvním sloupci všech 10 souborů."""
        all_sequences = []
        for data in train_data_dict.values():
            seqs = self._create_sequences(data)
            if seqs.size > 0:
                all_sequences.append(seqs)
        
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
        print(f"--- LSTM-AE Training Complete (Loss: {loss.item():.6f}) ---")

    def prediction(self, test_data_dict):
        """Vrací slovník množin outlierů pro každý kanál."""
        self.model.eval()
        outliers_dict = {}

        for cid, data in test_data_dict.items():
            X_test = self._create_sequences(data)
            if X_test.size == 0:
                outliers_dict[cid] = set()
                continue

            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                reconstructed = self.model(X_tensor)
                # Výpočet MSE pro každé okno
                mse = torch.mean((X_tensor - reconstructed)**2, dim=(1, 2)).cpu().numpy()
            
            # Mapování skóre na původní délku dat (padding nulami na začátek)
            full_scores = np.zeros(len(data))
            full_scores[self.seq_len - 1:] = mse
            
            # Dynamický threshold pro daný soubor (nebo globální, pokud bys chtěl)
            threshold = np.percentile(full_scores, self.percentile)
            predicted_indices = np.where(full_scores > threshold)[0]
            
            # Převod na set (množinu) podle tvého zadání
            outliers_dict[cid] = set(predicted_indices)
            print(f"Channel {cid}: Predicted {len(outliers_dict[cid])} outlier points.")

        return outliers_dict