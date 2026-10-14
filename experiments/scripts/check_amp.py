"""One forward/backward step of every model under bf16 autocast on the available device.

Run on a GPU node before submitting the full study (takes seconds):
  python scripts/check_amp.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from _common import config_from, parser
from clcpd.features.attention import attention_stats
from clcpd.features.graph import graph_features, lead_graph
from clcpd.models.clt import CLTransCPD, recon_error
from clcpd.models.cls_baselines import build_classifier
from clcpd.models.cpd_baselines import build_deep, _down
from clcpd.models.losses import contrastive_loss, masked_mse


def main():
    args = parser(__doc__).parse_args()
    cfg = config_from(args)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    B, L, T = 8, 12, int(cfg.segmentation.ref_len)
    x = torch.randn(4 * B, L, T, device=dev)
    m = torch.zeros(4 * B, T, dtype=torch.bool, device=dev)
    m[:, :200] = True
    model = CLTransCPD(cfg).to(dev)
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        out = model(x, m)
    z = out["z"].float()
    loss = contrastive_loss("triplet", z[:B], z[B:2 * B], z[2 * B:3 * B], 1.0, 0.1) \
        + masked_mse(out["recon"].float(), out["target"].float(), m) \
        + F.binary_cross_entropy_with_logits(out["logits"].float(), torch.rand(4 * B, 5, device=dev))
    loss.backward()
    print("CL-TransCPD train step ok, loss", float(loss))
    with torch.no_grad():
        o = model(x[:B], m[:B], need_weights=True)
        print("attention features", tuple(attention_stats(o["attn_t"], o["pvalid"]).shape),
              "graph features", tuple(graph_features(lead_graph(o["attn_l"], o["pvalid"])).shape),
              "recon error", tuple(recon_error(o["recon"], o["target"], m[:B]).shape))
    for name in cfg.cls_baselines.models:
        net = build_classifier(name).to(dev)
        xr = torch.randn(4, 12, 1000, device=dev)
        amp = name != "bilstm"
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            y = net(xr).float()
        y.sum().backward()
        print(f"{name}: ok {tuple(y.shape)}")
    w = int(round(cfg.cpd_baselines.window_s * cfg.data.fs))
    for name in ("tire", "lstm_ae", "tscp2", "ocae"):
        net = build_deep(name, 12, w).to(dev)
        xw = torch.randn(4, 12, w, device=dev)
        if name == "tire":
            y = net(_down(xw).flatten(1))[1]
        elif name == "lstm_ae":
            y = net(_down(xw).transpose(1, 2))
        else:
            y = net(xw)
            y = y[1] if isinstance(y, tuple) else y
        y.float().sum().backward()
        print(f"{name}: ok")
    print("ALL AMP CHECKS PASSED")


if __name__ == "__main__":
    main()
