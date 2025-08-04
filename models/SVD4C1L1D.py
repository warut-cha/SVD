from torch import nn


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
