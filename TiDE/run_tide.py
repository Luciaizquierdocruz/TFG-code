import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from TiDE import Model  # Asumo que tienes esta importación correcta

class Configs:
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.seq_len = 20
        self.label_len = 10
        self.pred_len = 10
        self.d_model = 64
        self.d_ff = 64
        self.e_layers = 2
        self.d_layers = 1
        self.freq = 'd'
        self.c_out = 1
        self.dropout = 0.2

class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len, label_len, pred_len):
        self.series = series
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    # Cambiamos __len__ para que las ventanas no se solapen
    def __len__(self):
        return (len(self.series) - self.seq_len - self.pred_len + 1) // self.pred_len

    # Cambiamos __getitem__ para que avance con step pred_len
    def __getitem__(self, idx):
        start = idx * self.pred_len
        x_enc = self.series[start : start + self.seq_len]
        x_mark_enc = np.zeros((self.seq_len, 3))  # variables exógenas dummy
        x_dec = self.series[start + self.seq_len - self.label_len : start + self.seq_len + self.pred_len]
        batch_y_mark = np.zeros((self.seq_len + self.pred_len, 3))
        y_true = self.series[start + self.seq_len : start + self.seq_len + self.pred_len]

        return {
            'x_enc': torch.tensor(x_enc, dtype=torch.float32).unsqueeze(-1),
            'x_mark_enc': torch.tensor(x_mark_enc, dtype=torch.float32),
            'x_dec': torch.tensor(x_dec, dtype=torch.float32).unsqueeze(-1),
            'batch_y_mark': torch.tensor(batch_y_mark, dtype=torch.float32),
            'y_true': torch.tensor(y_true, dtype=torch.float32).unsqueeze(-1),
        }

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x_enc = batch['x_enc'].to(device)
        x_mark_enc = batch['x_mark_enc'].to(device)
        x_dec = batch['x_dec'].to(device)
        batch_y_mark = batch['batch_y_mark'].to(device)
        y_true = batch['y_true'].to(device)

        optimizer.zero_grad()
        out = model(x_enc, x_mark_enc, x_dec, batch_y_mark)

        if out.dim() == 4:
            pred = out[:, :, 0, 0]
        elif out.dim() == 3:
            pred = out[:, :, 0]
        else:
            raise ValueError(f"Unexpected output dimension: {out.dim()}")

        loss = loss_fn(pred, y_true.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def predict_for_2024(model, series_train, series_test, configs, device):
    model.eval()
    preds = []

    full_series = np.concatenate([series_train, series_test])
    seq_len = configs.seq_len
    label_len = configs.label_len
    pred_len = configs.pred_len

    start_idx = len(series_train) - seq_len
    # Avanzamos con step = pred_len para no solapar predicciones
    for idx in range(start_idx, len(full_series) - seq_len - pred_len + 1, pred_len):
        x_enc = full_series[idx : idx + seq_len]
        x_mark_enc = np.zeros((seq_len, 3))
        x_dec = full_series[idx + seq_len - label_len : idx + seq_len + pred_len]
        batch_y_mark = np.zeros((seq_len + pred_len, 3))

        x_enc = torch.tensor(x_enc, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        x_mark_enc = torch.tensor(x_mark_enc, dtype=torch.float32).unsqueeze(0).to(device)
        x_dec = torch.tensor(x_dec, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        batch_y_mark = torch.tensor(batch_y_mark, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x_enc, x_mark_enc, x_dec, batch_y_mark)
            if out.dim() == 4:
                pred = out[:, :, 0, 0].cpu().numpy()
            elif out.dim() == 3:
                pred = out[:, :, 0].cpu().numpy()
            else:
                raise ValueError("Unexpected model output dimension")

            preds.extend(pred.flatten())

    # Cortar para que coincida con el tamaño del test
    preds = np.array(preds[:len(series_test)])
    return preds

def main(df_train, df_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    series_train = df_train['Close'].values.astype(np.float32)
    series_test = df_test['Close'].values.astype(np.float32)

    configs = Configs()
    model = Model(configs)
    model.to(device)

    dataset_train = TimeSeriesDataset(series_train, configs.seq_len, configs.label_len, configs.pred_len)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(100):
        loss = train_epoch(model, dataloader_train, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1} loss: {loss:.6f}")

    preds_2024 = predict_for_2024(model, series_train, series_test, configs, device)

    return preds_2024


