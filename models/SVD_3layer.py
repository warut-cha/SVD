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