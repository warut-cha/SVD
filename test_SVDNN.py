import torch
import numpy as np
from SVDNN import SVDNet

def test_model(model_path: str, test_data_path: str):
    # Define model architecture (same as training)
    model = SVDNet(M=64, N=64, r=32)  # Use same M, N, r as before
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Put in inference mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_np = np.load(test_data_path)  # shape: (B, M, N, 2)
    test_tensor = torch.from_numpy(test_np).float().to(device)

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

    # Save each to a separate .npz file
    np.savez("1.npz", U=U_np)
    np.savez("2.npz", s=s_np)
    np.savez("3.npz", V=V_np)


if __name__ == "__main__":
    model_path = 'SVDNet_model.pth'
    test_data_path = './CompetitionData1/Round1TestData1.npy'
    test_model(model_path, test_data_path)
