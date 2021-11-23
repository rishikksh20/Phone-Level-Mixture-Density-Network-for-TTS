import torch
import torch.nn as nn
from torch.nn import functional as F


class ProsodyExtractor(nn.Module):
    """ Prosody Extractor """

    def __init__(self, n_mel_channels, d_model, kernel_size):
        super(ProsodyExtractor, self).__init__()
        self.d_model = d_model
        self.conv_stack = nn.Sequential(
            ConvBlock2D(
                in_channels=1,
                out_channels=self.d_model,
                kernel_size=kernel_size,
            ),
            ConvBlock2D(
                in_channels=self.d_model,
                out_channels=1,
                kernel_size=kernel_size,
            ),
        )
        self.gru = nn.GRU(
            input_size=n_mel_channels,
            hidden_size=self.d_model,
            batch_first=True,
            bidirectional=True,
        )

    def get_prosody_embedding(self, mel):
        """
        mel -- [B, mel_len, n_mel_channels], B=1
        h_n -- [B, 2 * d_model], B=1
        """
        x = self.conv_stack(mel.unsqueeze(-1)).squeeze(-1)
        _, h_n = self.gru(x)
        h_n = torch.cat((h_n[0], h_n[1]), dim=-1)
        return h_n

    def forward(self, mel, mel_len, duration, src_len):
        """
        mel -- [B, mel_len, n_mel_channels]
        mel_len -- [B,]
        duration -- [B, src_len]
        src_len -- [B,]
        batch -- [B, src_len, 2 * d_model]
        """
        batch = []
        for m, m_l, d, s_l in zip(mel, mel_len, duration, src_len):
            b = []
            for m_p in torch.split(m[:m_l], list(d[:s_l].int()), dim=0):
                b.append(self.get_prosody_embedding(m_p.unsqueeze(0)).squeeze(0))
            batch.append(torch.stack(b, dim=0))

        return self.pad(batch)

    def pad(self, input_ele, mel_max_length=None):
        if mel_max_length:
            max_len = mel_max_length
        else:
            max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

        out_list = list()
        for i, batch in enumerate(input_ele):
            if len(batch.shape) == 1:
                one_batch_padded = F.pad(
                    batch, (0, max_len - batch.size(0)), "constant", 0.0
                )
            elif len(batch.shape) == 2:
                one_batch_padded = F.pad(
                    batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
                )
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded


class MDN(nn.Module):
    """ Mixture Density Network """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.w = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=-1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, x):
        """
        x -- [B, src_len, in_features]
        w -- [B, src_len, num_gaussians]
        sigma -- [B, src_len, num_gaussians, out_features]
        mu -- [B, src_len, num_gaussians, out_features]
        """
        B, src_len, _ = x.shape
        w = self.w(x)
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(B, src_len, self.num_gaussians, self.out_features)
        mu = self.mu(x)
        mu = mu.view(B, src_len, self.num_gaussians, self.out_features)
        return w, sigma, mu


class ProsodyPredictor(nn.Module):
    """ Prosody Predictor """

    def __init__(self, d_model, kernel_size, num_gaussians, dropout):
        super(ProsodyPredictor, self).__init__()
        self.d_model = d_model
        self.conv_stack = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.d_model,
                    out_channels=self.d_model,
                    kernel_size=kernel_size[i],
                    dropout=dropout,
                    normalization=nn.LayerNorm,
                    transpose=True,
                )
                for i in range(2)
            ]
        )
        self.gru_cell = nn.GRUCell(
            self.d_model + 2 * self.d_model,
            2 * self.d_model,
        )
        self.gmm_mdn = MDN(
            in_features=2 * self.d_model,
            out_features=2 * self.d_model,
            num_gaussians=num_gaussians,
        )

    def init_state(self, x):
        """
        x -- [B, src_len, d_model]
        p_0 -- [B, 2 * d_model]
        self.gru_hidden -- [B, 2 * d_model]
        """
        B, _, d_model = x.shape
        p_0 = torch.zeros((B, 2 * d_model), device=x.device, requires_grad=True)
        self.gru_hidden = torch.zeros((B, 2 * d_model), device=x.device, requires_grad=True)
        return p_0

    def forward(self, h_text, mask=None):
        """
        h_text -- [B, src_len, d_model]
        mask -- [B, src_len]
        outputs -- [B, src_len, 2 * d_model]
        """
        x = h_text
        for conv_layer in self.conv_stack:
            x = conv_layer(x, mask=mask)

        # Autoregressive Prediction
        p_0 = self.init_state(x)

        outputs = [p_0]
        for i in range(x.shape[1]):
            p_input = torch.cat((x[:, i], outputs[-1]), dim=-1) # [B, 3 * d_model]
            self.gru_hidden = self.gru_cell(p_input, self.gru_hidden) # [B, 2 * d_model]
            outputs.append(self.gru_hidden)
        outputs = torch.stack(outputs[1:], dim=1) # [B, src_len, 2 * d_model]

        # GMM-MDN
        w, sigma, mu = self.gmm_mdn(outputs)
        if mask is not None:
            w = w.masked_fill(mask.unsqueeze(-1), 0 if self.training else 1e-9) # 1e-9 for categorical sampling
            sigma = sigma.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
            mu = mu.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)

        return w, sigma, mu

    @staticmethod
    def sample(w, sigma, mu, mask=None):
        """ Draw samples from a GMM-MDN 
        w -- [B, src_len, num_gaussians]
        sigma -- [B, src_len, num_gaussians, out_features]
        mu -- [B, src_len, num_gaussians, out_features]
        mask -- [B, src_len]
        output -- [B, src_len, out_features]
        """
        from torch.distributions import Categorical
        batch = []
        for i in range(w.shape[1]):
            w_i, sigma_i, mu_i = w[:, i], sigma[:, i], mu[:, i]
            ws = Categorical(w_i).sample().view(w_i.size(0), 1, 1)
            # Choose a random sample, one randn for batch X output dims
            # Do a (output dims)X(batch size) tensor here, so the broadcast works in
            # the next step, but we have to transpose back.
            gaussian_noise = torch.randn(
                (sigma_i.size(2), sigma_i.size(0)), requires_grad=False).to(w.device)
            variance_samples = sigma_i.gather(1, ws).detach().squeeze()
            mean_samples = mu_i.detach().gather(1, ws).squeeze()
            batch.append((gaussian_noise * variance_samples + mean_samples).transpose(0, 1))
        output = torch.stack(batch, dim=1)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        return output




class ConvBlock(nn.Module):
    """ 1D Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=None, normalization=nn.BatchNorm1d, activation=nn.ReLU, transpose=False):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
                transpose=transpose
            ),
            normalization(out_channels),
            activation(),
        )
        self.dropout = dropout if dropout is not None else None
        self.transpose = transpose

    def forward(self, enc_input, mask=None):
        if not self.transpose:
            enc_input = enc_input.contiguous().transpose(1, 2)
        enc_output = self.conv_layer(enc_input)
        if self.dropout is not None:
            enc_output = F.dropout(enc_output, self.dropout, self.training)

        if not self.transpose:
            enc_output = enc_output.contiguous().transpose(1, 2)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class ConvBlock2D(nn.Module):
    """ 2D Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=None, normalization=nn.BatchNorm2d, activation=nn.ReLU, transpose=False):
        super(ConvBlock2D, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm2D(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, int((kernel_size - 1) / 2)),
                bias=False,
                w_init_gain="tanh",
                transpose=transpose,
            ),
            normalization(out_channels),
            activation(),
        )
        self.dropout = dropout if dropout is not None else None
        self.transpose = transpose

    def forward(self, enc_input, mask=None):
        """
        enc_input -- [B, H, W, C_in]
        mask -- [B, H]
        """
        if not self.transpose:
            enc_input = enc_input.contiguous().permute(0, 3, 1, 2) # [B, C_in, H, W]
        enc_output = self.conv_layer(enc_input)
        if self.dropout is not None:
            enc_output = F.dropout(enc_output, self.dropout, self.training)

        if not self.transpose:
            enc_output = enc_output.contiguous().permute(0, 2, 3, 1) # [B, H, W, C_out]
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)

        return enc_output


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        transpose=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        if self.transpose:
            x = x.contiguous().transpose(1, 2)

        return x


class ConvNorm2D(nn.Module):
    """ 2D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        transpose=False,
    ):
        super(ConvNorm2D, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        self.transpose = transpose

    def forward(self, x):
        """
        x -- [B, H, W, C] or [B, C, H, W]
        """
        if self.transpose:
            x = x.contiguous().permute(0, 3, 1, 2) # [B, C, H, W]
        x = self.conv(x)
        if self.transpose:
            x = x.contiguous().permute(0, 2, 3, 1) # [B, H, W, C]

        return x
