import torch
import numpy as np

from mac_calc import estimate_model_complexity
from torch import nn

from models import SVDNetPrune


def fix_rank(tensor, target_r, dim):
    current_r = tensor.shape[dim]
    if current_r > target_r:
        return tensor.narrow(dim, 0, target_r)  # Truncate
    elif current_r < target_r:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = target_r - current_r
        pad_tensor = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad_tensor], dim=dim)
    else:
        return tensor


def test_model(model: nn.Module, test_data_path: str, file_name: str, r: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.eval()

    test_np = np.load(test_data_path)  # shape: (B, M, N, 2)
    test_tensor = torch.from_numpy(test_np).float()

    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

    U_list, s_list, V_list = [], [], []

    with torch.no_grad():
        for (h_test,) in test_loader:
            h_test = h_test.to(device)
            U, s, V = model(h_test)
            # Pad or truncate to FIXED_R

            # U_fixed = fix_rank(U, FIXED_R, dim=2) # (B, M, FIXED_R, Q)
            # s_fixed = fix_rank(s, FIXED_R, dim=1) # (B, FIXED_R)
            # V_fixed = fix_rank(V, FIXED_R, dim=2) # (B, N, FIXED_R, Q)

            U_list.append(U.cpu())
            s_list.append(s.cpu())
            V_list.append(V.cpu())

    # Concatenate results
    U_all = torch.cat(U_list)
    s_all = torch.cat(s_list)
    V_all = torch.cat(V_list)

    # Convert to NumPy
    U_np = U_all.numpy()
    s_np = s_all.numpy()
    V_np = V_all.numpy()

    # Estimate model complexity
    mega_macs = estimate_model_complexity(model)
    print(f"Estimated MACs: {mega_macs:.2f} MegaMACs")

    # Save each to a separate .npz file
    np.savez(f"{file_name}.npz", U=U_np, S=s_np, V=V_np, C=mega_macs)


if __name__ == "__main__":
    model_path = 'SVDNet_model.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVDNetPrune(M=64, N=64, r=16).to(device)  # Use same M, N, r as before
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    for i in range(1, 4):
        test_data_path = f'./CompetitionData1/Round1TestData{i}.npy'
        test_model(model, test_data_path, f"submission/{i}")
