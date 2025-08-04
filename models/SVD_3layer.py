from torch import nn

class SVDNet3Layer(nn.Module):
    def __init__(self, M, N, r):
        super(SVDNet3Layer, self).__init__()
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
            nn.BatchNorm2d(first_conv_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=first_conv_out_channels, out_channels=second_conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(second_conv_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=second_conv_out_channels, out_channels=third_conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(third_conv_out_channels),
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



class SVDNet3C2L(nn.Module):
    def __init__(self, M, N, r):
        super(SVDNet3C2L, self).__init__()
        self.M = M
        self.N = N
        self.r = r

        first_conv_out_channels = 4
        second_conv_out_channels = 8
        third_conv_out_channels = 16

        m_after_conv = M // 8
        n_after_conv = N // 8
        size_after_conv = third_conv_out_channels * m_after_conv * n_after_conv

        size_after_fc_1 = size_after_conv // 2
        size_after_fc_2 = size_after_fc_1 // 2

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
            nn.Linear(size_after_fc_1, size_after_fc_2),
            nn.ReLU(),
        )

        in_size= size_after_fc_2
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


class SVDNetAvg(nn.Module):
    def __init__(self, M, N, r):
        super(SVDNetAvg, self).__init__()
        self.M = M
        self.N = N
        self.r = r

        first_conv_out_channels = 2
        second_conv_out_channels = 4
        third_conv_out_channels = 8

        m_after_conv = M // 8
        n_after_conv = N // 8
        size_after_conv = third_conv_out_channels * m_after_conv * n_after_conv

        size_after_fc_1 = size_after_conv // 2
        size_after_fc_2 = size_after_fc_1 // 2

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=first_conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(first_conv_out_channels),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(in_channels=first_conv_out_channels, out_channels=second_conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(second_conv_out_channels),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(in_channels=second_conv_out_channels, out_channels=third_conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(third_conv_out_channels),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Flatten()
        )

        self.fc_hidden = nn.Sequential(
            nn.Linear(size_after_conv, size_after_fc_1),
            nn.ReLU(),
            nn.Linear(size_after_fc_1, size_after_fc_2),
            nn.ReLU(),
        )

        in_size= size_after_fc_2
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


class SVDNetDeep(nn.Module):
    def __init__(self, M, N, r):
        super(SVDNetDeep, self).__init__()
        self.M = M
        self.N = N
        self.r = r

        m_after_conv = M // 4
        n_after_conv = N // 4
        size_after_conv = 8 * m_after_conv * n_after_conv

        size_after_fc_1 = size_after_conv // 2
        size_after_fc_2 = size_after_fc_1 // 2

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        self.fc_hidden = nn.Sequential(
            nn.Linear(size_after_conv, size_after_fc_1),
            nn.ReLU(),
            nn.Dropout(p=0.4),
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


class SVDNetDeeppp(nn.Module):
    def __init__(self, M, N, r):
        super(SVDNetDeeppp, self).__init__()
        self.M = M  # Number of receive antennas (64)
        self.N = N  # Number of transmit antennas (64)
        self.r = r  # Rank of SVD approximation

        # Define a smaller convolutional backbone
        self.backbone = nn.Sequential(
            # Conv1: 16 filters, input channels=2 (real and imaginary parts)
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample: 64x64 -> 32x32
            # Conv2: 32 filters
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample: 32x32 -> 16x16
            # Conv3: 64 filters
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate size after convolutional layers
        # After two max-pooling layers (64x64 -> 32x32 -> 16x16), with 64 filters
        size_after_conv = 64 * (M // 4) * (N // 4)  # M//4 = 16, N//4 = 16 -> 64 * 16 * 16 = 16384
        size_after_fc = size_after_conv // 4  # Reduce to 4096 for FC layer

        # Single FC layer with dropout
        self.fc_hidden = nn.Sequential(
            nn.Linear(size_after_conv, size_after_fc),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        # Output heads for U, s, V
        self.u_head = nn.Linear(size_after_fc, M * r * 2)  # U: (M, r, 2) for real/imag
        self.s_head = nn.Linear(size_after_fc, r)         # s: (r,)
        self.v_head = nn.Linear(size_after_fc, N * r * 2)  # V: (N, r, 2) for real/imag

        # Activation for singular values (ensure non-negative)
        self.s_activation = nn.Softplus()

    def forward(self, x):
        # Input: (batch_size, 64, 64, 2)
        x = x.permute(0, 3, 1, 2)  # -> (batch_size, 2, 64, 64)

        # Pass through convolutional backbone
        features = self.backbone(x)  # -> (batch_size, size_after_conv)

        # Pass through fully connected layer
        hidden = self.fc_hidden(features)  # -> (batch_size, size_after_fc)

        # Output heads
        u_flat = self.u_head(hidden)  # -> (batch_size, M * r * 2)
        U = u_flat.view(-1, self.M, self.r, 2)  # -> (batch_size, M, r, 2)

        s_raw = self.s_head(hidden)  # -> (batch_size, r)
        s = self.s_activation(s_raw)  # Ensure non-negative singular values

        v_flat = self.v_head(hidden)  # -> (batch_size, N * r * 2)
        V = v_flat.view(-1, self.N, self.r, 2)  # -> (batch_size, N, r, 2)

        return U, s, V
