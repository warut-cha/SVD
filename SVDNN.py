import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from models.SVD_3layer import SVDNet3C2L, SVDNet3Layer, SVDNetAvg, SVDNetDeep
from models.SVD_prune import SVDNetPrune
from test_SVDNN import test_model


BATCH_SIZE = 64
TEST_RATIO = 0.2
NUM_EPOCHS = 30
LEARNING_RATE = 5e-4
M, N, Q, r = 64, 64, 2, 24

@torch.jit.script
def complex_matmul_fused(A, B):
    """
    A: (batch, M, K, 2)   where A[...,0]=real, A[...,1]=imag
    B: (batch, K, N, 2)
    returns: (batch, M, N, 2)
    """
    A_r, A_i = A[..., 0], A[..., 1]       # (batch,M,K)
    B_r, B_i = B[..., 0], B[..., 1]       # (batch,K,N)

    top_A = torch.cat([ A_r, -A_i ], dim=2)  # (batch, M, 2K)
    bot_A = torch.cat([ A_i,  A_r ], dim=2)  # (batch, M, 2K)
    Ar    = torch.cat([ top_A, bot_A ], dim=1)  # (batch, 2M, 2K)

    top_B = torch.cat([ B_r, -B_i ], dim=2)  # (batch, K, 2N)
    bot_B = torch.cat([ B_i,  B_r ], dim=2)  # (batch, K, 2N)
    Br    = torch.cat([ top_B, bot_B ], dim=1)  # (batch, 2K, 2N)

    C = torch.bmm(Ar, Br)                    # (batch, 2M, 2N)

    M = A.shape[1]
    N = B.shape[2]
    real = C[:, :M, :N]
    imag = C[:, M:, :N]

    return torch.stack([real, imag], dim=-1)  # (batch, M, N, 2)

@torch.jit.script
def complex_matmul(A, B):
    real_part = A[..., 0] @ B[..., 0] - A[..., 1] @ B[..., 1]
    imag_part = A[..., 0] @ B[..., 1] + A[..., 1] @ B[..., 0]
    return torch.stack([real_part, imag_part], dim=-1)


def complex_hermitian(A):
    return torch.stack([A[..., 0].transpose(-2, -1), -A[..., 1].transpose(-2, -1)], dim=-1)

class SVDNet(nn.Module):
    def __init__(self, M, N, r):
        super(SVDNet, self).__init__()
        self.M = M
        self.N = N
        self.r = r

        first_conv_out_channels = 8
        second_conv_out_channels = 32
        m_after_conv = M // 4
        n_after_conv = N // 4
        size_after_conv = second_conv_out_channels * m_after_conv * n_after_conv
        size_after_fc_1 = size_after_conv // 2
        size_after_fc_2 = size_after_fc_1 // 2

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=first_conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=first_conv_out_channels, out_channels=second_conv_out_channels, kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        self.fc_hidden = nn.Sequential(
            nn.Linear(size_after_conv, size_after_fc_1),
            nn.ReLU(),
            nn.Linear(size_after_fc_1, size_after_fc_2),
            nn.ReLU(),
        )

        in_size = size_after_fc_2
        self.u_head = nn.Linear(in_size, M * r * 2)
        self.s_head = nn.Linear(in_size, r)
        self.v_head = nn.Linear(in_size, N * r * 2)

        self.s_activation = nn.Softplus()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # -> (batch, 2, M, N)

        features = self.backbone(x)
        hidden = self.fc_hidden(features)

        u_flat = self.u_head(hidden)
        U = u_flat.view(-1, self.M, self.r, 2)

        s_raw = self.s_head(hidden)
        s = self.s_activation(s_raw)

        v_flat = self.v_head(hidden)
        V = v_flat.view(-1, self.N, self.r, 2)

        return U, s, V


class NormalizedChannelDataset(Dataset):
    def __init__(self, data, labels, mean, std):
        self.data = data
        self.labels = labels
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the raw data and label
        x = self.data[idx]
        y = self.labels[idx]

        # Apply the pre-calculated normalization
        x_normalized = (x - self.mean) / (self.std + 1e-8)

        # Convert to tensors
        return torch.from_numpy(x_normalized).float(), torch.from_numpy(y).float()


class ApproximationErrorLoss(nn.Module):
    def __init__(self):
        super(ApproximationErrorLoss, self).__init__()


    def singular_value_order_penalty(self, s: torch.Tensor) -> torch.Tensor:
        """
        Penalizes non-monotonic singular values. s shape: (batch, r)
        """
        diff = s[:, :-1] - s[:, 1:]  # Should be ≥ 0 if properly ordered
        penalty = torch.relu(-diff)  # Penalize if s[i] < s[i+1]
        return penalty.mean()


    def forward(self, U, s, V, H_label):
        # H_label is of shape (batch, M, N, 2)
        # U is (batch, M, r, 2), s is (batch, r), V is (batch, N, r, 2)
        batch_size, M, r, _ = U.shape
        N = V.shape[1]

        # Reconstruction Loss: || H_label - U * Sigma * V^H ||
        Sigma = torch.diag_embed(s)  # Shape: (batch, r, r)
        Sigma_complex = torch.stack([Sigma, torch.zeros_like(Sigma)], dim=-1)  # (batch, r, r, 2)

        V_H = complex_hermitian(V)  # Shape: (batch, r, N, 2)
        U_Sigma = complex_matmul_fused(U, Sigma_complex)  # Shape: (batch, M, r, 2)
        H_pred = complex_matmul_fused(U_Sigma, V_H)  # Shape: (batch, M, N, 2)

        h_flat = H_label.view(batch_size, -1)
        diff_flat = (H_label - H_pred).view(batch_size, -1)

        norm_h_label = torch.norm(h_flat, p=2, dim=1)
        norm_diff = torch.norm(diff_flat, p=2, dim=1)

        reconstruction_loss = (norm_diff / (norm_h_label + 1e-8)).mean()

        # Orthogonality Loss
        U_H = complex_hermitian(U)  # (batch, r, M, 2)
        U_H_U = complex_matmul_fused(U_H, U)  # (batch, r, r, 2)

        V_H_V = complex_matmul_fused(V_H, V)  # (batch, r, r, 2)

        I_r = torch.eye(r, device=U.device).expand(batch_size, r, r)
        I_r_complex = torch.stack([I_r, torch.zeros_like(I_r)], dim=-1)

        u_err_flat = (U_H_U - I_r_complex).view(batch_size, -1)
        v_err_flat = (V_H_V - I_r_complex).view(batch_size, -1)

        ortho_loss_U = torch.norm(u_err_flat, p=2, dim=1).mean()
        ortho_loss_V = torch.norm(v_err_flat, p=2, dim=1).mean()

        # order_penalty = self.singular_value_order_penalty(s)

        total_loss = reconstruction_loss + ortho_loss_U + ortho_loss_V
        return total_loss


def split_dataset(x_np, y_np, test_size=0.2, random_state=42):
    """
    Splits numpy arrays into PyTorch TensorDatasets for training and testing.

    Args:
        x_np: numpy array of input data
        y_np: numpy array of labels
        test_size: fraction for test set
        random_state: reproducibility seed

    Returns:
        train_dataset, test_dataset (both TensorDataset)
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x_np, y_np, test_size=test_size, random_state=random_state
    )

    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    return train_dataset, test_dataset


def load_combined_data():
    train_list = []
    label_list = []

    for i in range(1, 4):
        train_np = np.load(f'./CompetitionData1/Round1TrainData{i}.npy')
        label_np = np.load(f'./CompetitionData1/Round1TrainLabel{i}.npy')
        train_list.append(train_np)
        label_list.append(label_np)

    combined_train = np.concatenate(train_list, axis=0)  # Concatenate along batch dim
    combined_label = np.concatenate(label_list, axis=0)

    return combined_train, combined_label


def train(model: nn.Module, test_data_path: str, label_data_path: str, model_name='SVDNet'):
    """
    Trains the SVDNet model.

    Args:
        model (nn.Module): The model to train.

    Example:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = SVDNet(M=M, N=N, r=r).to(device)
        train(model)
    """
    device = next(model.parameters()).device

    x = 1
    train_np = np.load(test_data_path)  # y = Hr + noise
    label_np = np.load(label_data_path)
    # all_data_np, all_labels_np = load_combined_data()

    # 2. Split into training and testing sets
    train_np, test_np, train_labels_np, test_labels_np = train_test_split(
        train_np, label_np, test_size=TEST_RATIO
    )

    mean = np.mean(train_np)
    std = np.std(train_np)

    train_dataset = NormalizedChannelDataset(train_np, train_labels_np, mean, std)
    test_dataset = NormalizedChannelDataset(test_np, test_labels_np, mean, std)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    loss_fn = ApproximationErrorLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS + 1)

    best_loss = float('inf')
    best_state_dict = None

    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (h_noise, h_label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                          desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            h_noise = h_noise.to(device)
            h_label = h_label.to(device)

            optimizer.zero_grad()
            U_pred, s_pred, V_pred = model(h_noise)
            loss = loss_fn(U_pred, s_pred, V_pred, h_label)

            loss.backward()
            optimizer.step()

        # Evaluate on test set
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for h_noise, h_label in test_loader:
                h_noise = h_noise.to(device)
                h_label = h_label.to(device)
                U_pred, s_pred, V_pred = model(h_noise)
                test_loss = loss_fn(U_pred, s_pred, V_pred, h_label)
                total_test_loss += test_loss.item()

        mean_test_loss = total_test_loss / len(test_loader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Test Loss: {mean_test_loss:.4f}")

        # Save best model
        if mean_test_loss < best_loss:
            best_loss = mean_test_loss
            best_state_dict = model.state_dict()  # Shallow copy (fine for PyTorch)

    print("--- Training Finished ---")
    print(f"✅ Best Mean Test Loss: {best_loss:.4f}")

    scheduler.step()
    # Restore best model and save it
    model.load_state_dict(state_dict=best_state_dict)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = 'submission/SVDNet3C2L.pth'
    model = SVDNetDeep(M=M, N=N, r=r).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model_path = 'SVDNet_model.pth'
    model = SVDNet3LayerD(M=M, N=N, r=r).to(device)
    train(model,
          test_data_path=f'./CompetitionData1/Round1TrainData1.npy',
          label_data_path=f'./CompetitionData1/Round1TrainLabel1.npy',
          model_name='SVDNetDeep2')
    torch.save(model.state_dict(), 'submission/SVDNetDeep2.pth')
    for i in range(1, 4):
        test_data_path = f'./CompetitionData1/Round1TestData{i}.npy'
        test_model(model, test_data_path, f"submission/{i}", r=r)

