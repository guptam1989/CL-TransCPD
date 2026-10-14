"""Hybrid Time-Frequency Preprocessing (HTFP): a learned soft mask on the STFT magnitude."""
from __future__ import annotations

import torch
import torch.nn as nn


class HTFP(nn.Module):
    def __init__(self, n_fft: int = 64, hop: int = 16, channels: int = 16):
        super().__init__()
        self.n_fft, self.hop = int(n_fft), int(hop)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        c = int(channels)
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c, 1, 3, padding=1),
        )
        # start close to the identity mapping (sigmoid(5) ~ 0.993)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 5.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x: (B, L, T); mask: (B, T) bool of valid samples. Returns (filtered x, spectral mask)."""
        B, L, T = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float().reshape(B * L, T)
            spec = torch.stft(xf, self.n_fft, self.hop, window=self.window, center=True,
                              return_complex=True)
            gate = torch.sigmoid(self.net(torch.log1p(spec.abs()).unsqueeze(1))).squeeze(1)
            y = torch.istft(spec * gate, self.n_fft, self.hop, window=self.window, center=True, length=T)
        y = y.reshape(B, L, T)
        if mask is not None:
            y = y * mask[:, None, :].to(y.dtype)
        return y.to(x.dtype), gate
