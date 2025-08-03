import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split


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
        second_conv_out_channels = 16
        third_conv_out_channels = 32

        m_after_conv = M // 8
        n_after_conv = N // 8
        size_after_conv = third_conv_out_channels * m_after_conv * n_after_conv

        size_after_fc_1 = size_after_conv // 2

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=first_conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=first_conv_out_channels, out_channels=second_conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=second_conv_out_channels, out_channels=third_conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        self.fc_hidden = nn.Sequential(
            nn.Linear(size_after_conv, size_after_fc_1),
            nn.ReLU(),
        )

        in_size= size_after_fc_1
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


class ApproximationErrorLoss(nn.Module):
    def __init__(self):
        super(ApproximationErrorLoss, self).__init__()

    def forward(self, U, s, V, H_label):
        # H_label is of shape (batch, M, N, 2)
        # U is (batch, M, r, 2), s is (batch, r), V is (batch, N, r, 2)
        batch_size, M, r, _ = U.shape
        N = V.shape[1]

        # Reconstruction Loss: || H_label - U * Sigma * V^H ||
        Sigma = torch.diag_embed(s)  # Shape: (batch, r, r)
        Sigma_complex = torch.stack([Sigma, torch.zeros_like(Sigma)], dim=-1)  # (batch, r, r, 2)

        V_H = complex_hermitian(V)  # Shape: (batch, r, N, 2)
        U_Sigma = complex_matmul(U, Sigma_complex)  # Shape: (batch, M, r, 2)
        H_pred = complex_matmul(U_Sigma, V_H)  # Shape: (batch, M, N, 2)

        h_flat = H_label.view(batch_size, -1)
        diff_flat = (H_label - H_pred).view(batch_size, -1)

        norm_h_label = torch.norm(h_flat,    p=2, dim=1)
        norm_diff = torch.norm(diff_flat, p=2, dim=1)

        reconstruction_loss = (norm_diff / (norm_h_label + 1e-8)).mean()

        # Orthogonality Loss
        U_H = complex_hermitian(U)  # (batch, r, M, 2)
        U_H_U = complex_matmul(U_H, U)  # (batch, r, r, 2)

        V_H_V = complex_matmul(V_H, V)  # (batch, r, r, 2)

        I_r = torch.eye(r, device=U.device).expand(batch_size, r, r)
        I_r_complex = torch.stack([I_r, torch.zeros_like(I_r)], dim=-1)

        u_err_flat = (U_H_U - I_r_complex).view(batch_size, -1)
        v_err_flat = (V_H_V - I_r_complex).view(batch_size, -1)

        ortho_loss_U = torch.norm(u_err_flat, p=2, dim=1).mean()
        ortho_loss_V = torch.norm(v_err_flat, p=2, dim=1).mean()

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


def train():
    BATCH_SIZE = 64
    TEST_RATIO = 0.2
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    M, N, Q, r = 64, 64, 2, 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x = 1
    train_np = np.load(f'./CompetitionData1/Round1TrainData1.npy')  # y = Hr + noise
    label_np = np.load(f'./CompetitionData1/Round1TrainLabel1.npy')
    train = torch.from_numpy(train_np).float()
    label = torch.from_numpy(label_np).float()

    train_dataset, test_dataset = split_dataset(train_np, label_np, test_size=TEST_RATIO)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = SVDNet(M=M, N=N, r=r).to(device)
    loss_fn = ApproximationErrorLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        for i, (h_noise, h_label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            h_noise = h_noise.to(device)
            h_label = h_label.to(device)
            optimizer.zero_grad()

            U_pred, s_pred, V_pred = model(h_noise)

            loss = loss_fn(U_pred, s_pred, V_pred, h_label)

            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

    print("--- Training Finished ---")
    torch.save(model.state_dict(), 'SVDNet_model.pth')

    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for h_noise, h_label in test_loader:
            h_noise = h_noise.to(device)
            h_label = h_label.to(device)
            U_pred, s_pred, V_pred = model(h_noise)
            loss = loss_fn(U_pred, s_pred, V_pred, h_label)
            total_test_loss += loss.item()

    print(f"✅ Mean Test Loss: {total_test_loss / len(test_loader):.4f}")


if __name__ == "__main__":
    train()
