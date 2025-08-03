import torch
import numpy as np

from SVDNN import SVDNet
from mac_calc import estimate_model_complexity


def test_model(model_path: str, test_data_path: str, file_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SVDNet(M=64, N=64, r=32).to(device)  # Use same M, N, r as before
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
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

    # Save each to a separate .npz file
    np.savez(f"{file_name}.npz", U=U_np, S=s_np, V=V_np, C=mega_macs)


if __name__ == "__main__":
    model_path = 'SVDNet_model.pth'
    test_data_path = './CompetitionData1/Round1TestData1.npy'
    test_model(model_path, test_data_path, "submission/1")
    test_data_path = './CompetitionData1/Round1TestData2.npy'
    test_model(model_path, test_data_path, "submission/2")
    test_data_path = './CompetitionData1/Round1TestData3.npy'
    test_model(model_path, test_data_path, "submission/3")
