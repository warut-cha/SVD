import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm


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

        first_conv_out_channels = 16
        second_conv_out_channels = 32
        m_after_conv = M // 4
        n_after_conv = N // 4
        size_after_conv = second_conv_out_channels * m_after_conv * n_after_conv

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        self.u_head = nn.Linear(size_after_conv, M * r * 2)
        self.s_head = nn.Linear(size_after_conv, r)
        self.v_head = nn.Linear(size_after_conv, N * r * 2)

        self.s_activation = nn.Softplus()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # -> (batch, 2, M, N)

        features = self.backbone(x)

        u_flat = self.u_head(features)
        U = u_flat.view(-1, self.M, self.r, 2)

        s_raw = self.s_head(features)
        s = self.s_activation(s_raw)

        v_flat = self.v_head(features)
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


def train():
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")  # Force CPU for compatibility
    print(f"Using device: {device}")

    M, N, Q, r = 64, 64, 2, 32

    x = 1
    train_np = np.load(f'./CompetitionData1/Round1TrainData1.npy')  # y = Hr + noise
    label_np = np.load(f'./CompetitionData1/Round1TrainLabel1.npy')
    train = torch.from_numpy(train_np).float()
    label = torch.from_numpy(label_np).float()

    dataset = torch.utils.data.TensorDataset(train, label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SVDNet(M=M, N=N, r=r).to(device)
    loss_fn = ApproximationErrorLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    print("--- Starting Training ---")
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (h_noise, h_label) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}"):
            h_noise = h_noise.to(device)
            h_label = h_label.to(device)
            optimizer.zero_grad()

            U_pred, s_pred, V_pred = model(h_noise)

            loss = loss_fn(U_pred, s_pred, V_pred, h_label)

            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("--- Training Finished ---")
    torch.save(model.state_dict(), 'SVDNet_model.pth')


if __name__ == "__main__":
    train()
