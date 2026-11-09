"""TNN baseline with the hyperparameter optimum of the TNN paper (2026-11-09, revision; CBP recipe unchanged).

Same as scripts/run_tnn_official.py (authors' public PyTorch code, MIT License, Copyright (c) 2021 Wilhelm
Kirchgaessner; https://github.com/wkirgsn/thermal-nn; licence text in third_party/thermal-nn/LICENSE), with the sub-networks and optimiser set to the optimum of
Table 2 of Kirchgaessner, Wallscheid, Boecker, Eng. Appl. Artif. Intell. 117 (2023) 105537:
  power-loss net pi:   3 layers, 109 -> 3 -> outputs, ReLU -> sigmoid -> sigmoid, l2 1e-8 -> 1e-8
  conductance net gam: 3 layers, 2 -> 2 -> outputs, tanh -> sinus -> biased ELU,  l2 5.5e-6 -> 5.3e-8
  Nadam, initial learning rate 11e-3, TBPTT length 1227 samples, no subsequence splitting, no dedicated branches.
Interpretation choices (Table 2 does not fix them): three layers = two hidden layers + output layer; the two l2 rates
are applied to the first layer and to the later layers; biased ELU = ELU(x) + 1 as in the authors' repository
(NeuralOrdinaryDifferentialEq_torch.ipynb); sinus = sin(x). Inputs, normalisation, epochs (100, learning rate halved
at epoch 75) and the constant learnable capacitances are those of the public notebook, as in run_tnn_official.py.
Resumable: a checkpoint is written after every epoch; a finished run (meta file present) is skipped.
--mode cv --fold k | --mode test   --seed s   --threads n   --out DIR
"""
from __future__ import annotations
import sys, json, time, argparse, platform, os
from pathlib import Path
from typing import List, Tuple
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.nn import Parameter as TorchParam
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from pmsm_softsense import realdata

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["cv", "test"]); ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device("cpu")  # the notebook recommends CPU

out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "source_snapshot").mkdir(exist_ok=True)
(out_dir / "source_snapshot" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
tag = f"cv_f{args.fold}_s{args.seed}" if args.mode == "cv" else f"test_s{args.seed}"
if (out_dir / f"meta_{tag}.json").exists():
    print(tag, "already finished"); sys.exit(0)

# ---------------- data (same units as CBP) ----------------
M = realdata.manifest(); FOLDS = M["dev_cv5_over_non_G"]; DEV = sorted(p for f in FOLDS for p in f)
TEST = sorted(M["primary_protocol"]["test_locked_G"])
held = sorted(FOLDS[args.fold]) if args.mode == "cv" else TEST
train = [p for p in DEV if p not in held] if args.mode == "cv" else DEV
_ = realdata.load(train + held, final_evaluation=(args.mode == "test"))      # access gate of the project (locked test)
F = realdata._frame(); F = F[F.profile_id.isin(train + held)].copy()

target_cols = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]
temperature_cols = target_cols + ["ambient", "coolant"]
non_temperature_cols = [c for c in F.columns if c not in temperature_cols + ["profile_id"]]
F.loc[:, temperature_cols] /= 200  # deg C
scale = F.loc[F.profile_id.isin(train), non_temperature_cols].abs().max(axis=0)
F.loc[:, non_temperature_cols] /= scale
F = F.assign(i_s=lambda x: np.sqrt(x["i_d"] ** 2 + x["i_q"] ** 2), u_s=lambda x: np.sqrt(x["u_d"] ** 2 + x["u_q"] ** 2))
input_cols = [c for c in F.columns if c not in target_cols + ["profile_id"]]
F = F.loc[:, input_cols + ["profile_id"] + target_cols]
profile_sizes = F.groupby("profile_id").size()


def generate_tensor(profiles_list):
    """As in the notebook: time-major (T, B, F) tensor and masking sample weights."""
    tensor = np.full((profile_sizes[profiles_list].max(), len(profiles_list), F.shape[1] - 1), np.nan)
    for i, pid in enumerate(profiles_list):
        df = F[F.profile_id == pid]
        tensor[: len(df), i, :] = df.drop(columns="profile_id").to_numpy()
    sample_weights = 1 - np.isnan(tensor[:, :, 0])
    tensor = np.nan_to_num(tensor).astype(np.float32)
    return torch.from_numpy(tensor).to(device), torch.from_numpy(sample_weights).to(device)


# ---------------- model: notebook code, sub-networks per Table 2 of the TNN paper ----------------
class Biased_Elu(nn.Module):
    """From the authors' repository (NeuralOrdinaryDifferentialEq_torch.ipynb)."""
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1


class Sinus(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class DiffEqLayer(nn.Module):
    """This class is a container for the computation logic in each step.
    This layer could be used for any 'cell', also RNNs, LSTMs or GRUs."""

    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    def forward(self, input: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class TNNCell(nn.Module):
    """The main TNN logic (notebook), with the Table 2 optimum for the conductance and power-loss sub-NNs."""

    def __init__(self):
        super().__init__()
        self.sample_time = 0.5  # in s
        self.output_size = len(target_cols)
        self.caps = TorchParam(torch.Tensor(self.output_size).to(device))
        nn.init.normal_(self.caps, mean=-9.2, std=0.5)  # hand-picked init mean, might be application-dependent
        n_temps = len(temperature_cols)  # number of temperatures (targets and input)
        n_conds = int(0.5 * n_temps * (n_temps - 1))  # number of thermal conductances
        n_in = len(input_cols) + self.output_size
        # conductance net sub-NN (gamma): 2 -> 2 -> n_conds, tanh -> sinus -> biased ELU
        self.conductance_net = nn.Sequential(nn.Linear(n_in, 2), nn.Tanh(), nn.Linear(2, 2), Sinus(),
                                             nn.Linear(2, n_conds), Biased_Elu())
        # populate adjacency matrix. It is used for indexing the conductance sub-NN output
        self.adj_mat = np.zeros((n_temps, n_temps), dtype=int)
        adj_idx_arr = np.ones_like(self.adj_mat)
        triu_idx = np.triu_indices(n_temps, 1)
        adj_idx_arr = adj_idx_arr[triu_idx].ravel()
        self.adj_mat[triu_idx] = np.cumsum(adj_idx_arr) - 1
        self.adj_mat += self.adj_mat.T
        self.adj_mat = torch.from_numpy(self.adj_mat[: self.output_size, :]).type(torch.int64)  # crop
        self.n_temps = n_temps
        # power loss sub-NN (pi): 109 -> 3 -> outputs, ReLU -> sigmoid -> sigmoid
        self.ploss = nn.Sequential(nn.Linear(n_in, 109), nn.ReLU(), nn.Linear(109, 3), nn.Sigmoid(),
                                   nn.Linear(3, self.output_size), nn.Sigmoid())
        self.temp_idcs = [i for i, x in enumerate(input_cols) if x in temperature_cols]
        self.nontemp_idcs = [i for i, x in enumerate(input_cols) if x not in temperature_cols + ["profile_id"]]

    def forward(self, inp: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        prev_out = hidden
        temps = torch.cat([prev_out, inp[:, self.temp_idcs]], dim=1)
        sub_nn_inp = torch.cat([inp, prev_out], dim=1)
        conducts = torch.abs(self.conductance_net(sub_nn_inp))
        power_loss = torch.abs(self.ploss(sub_nn_inp))
        temp_diffs = torch.sum((temps.unsqueeze(1) - prev_out.unsqueeze(-1)) * conducts[:, self.adj_mat], dim=-1)
        out = prev_out + self.sample_time * torch.exp(self.caps) * (temp_diffs + power_loss)
        return prev_out, torch.clip(out, -1, 5)


t0 = time.time()
train_tensor, train_sample_weights = generate_tensor(train)
try:
    model = torch.jit.script(DiffEqLayer(TNNCell).to(device)); jit = True
except Exception as e:  # identical computation without scripting
    model = DiffEqLayer(TNNCell).to(device); jit = False; print("jit failed:", e)
n_params = int(sum(p.numel() for p in model.parameters()))
# l2 rates per layer (Table 2): first layer, later layers
L2 = []
for net, (a_first, a_rest) in [(model.cell.ploss, (1e-8, 1e-8)), (model.cell.conductance_net, (5.5e-6, 5.3e-8))]:
    lins = [m for m in net if isinstance(m, nn.Linear)] if not jit else \
        [getattr(net, n) for n in ["0", "2", "4"]]
    L2 += [(lins[0].weight, a_first)] + [(l.weight, a_rest) for l in lins[1:]]
loss_func = nn.MSELoss(reduction="none")
opt = optim.NAdam(model.parameters(), lr=11e-3)
n_epochs = args.epochs
tbptt_size = 1227
n_batches = np.ceil(train_tensor.shape[0] / tbptt_size).astype(int)

ckpt = out_dir / f"ckpt_{tag}.pt"; logp = out_dir / f"log_{tag}.txt"
start_epoch, rec = 0, None
if ckpt.exists():  # resume
    c = torch.load(ckpt, weights_only=False)
    model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); torch.set_rng_state(c["rng"])
    start_epoch, rec = c["epoch"] + 1, c["rec"]
    print(tag, "resumed at epoch", start_epoch, flush=True)
logf = open(logp, "a")
for epoch in range(start_epoch, n_epochs):
    hidden = train_tensor[0, :, -len(target_cols):]
    ep_sum, ep_w = 0.0, 0.0
    for i in range(n_batches):
        model.zero_grad()
        output, hidden = model(train_tensor[i * tbptt_size:(i + 1) * tbptt_size, :, : len(input_cols)], hidden.detach())
        loss = loss_func(output, train_tensor[i * tbptt_size:(i + 1) * tbptt_size, :, -len(target_cols):])
        sw = train_sample_weights[i * tbptt_size:(i + 1) * tbptt_size, :]
        ep_sum += float((loss.detach() * sw[:, :, None]).sum()); ep_w += float(sw.sum()) * len(target_cols)
        loss = (loss * sw[:, :, None] / sw.sum()).sum().mean() + sum(a * (w ** 2).sum() for w, a in L2)
        loss.backward()
        opt.step()
    if epoch == 75:
        for group in opt.param_groups:
            group["lr"] *= 0.5
    rec = {"epoch": epoch, "train_mse_K2": ep_sum / ep_w * 200 ** 2, "t": round(time.time() - t0)}
    logf.write(json.dumps(rec) + "\n"); logf.flush(); print(tag, rec, flush=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "rng": torch.get_rng_state(), "epoch": epoch,
                "rec": rec}, str(ckpt) + ".tmp")
    os.replace(str(ckpt) + ".tmp", ckpt)
final_train = rec["train_mse_K2"]
torch.save(model.state_dict(), out_dir / f"model_{tag}.pt")

# ---------------- evaluation (identical to run_tnn_official.py) ----------------
model.eval()
idx_amb = input_cols.index("ambient")
cont, rows = {}, []
TS = 0.5; TIMES = [0, 2, 4, 5, 10, 20, 30]; HOR = int(31 * 60 / TS)
def events(n): return list(range(0, n - int(5 * 60 / TS), int(15 * 60 / TS)))
with torch.no_grad():
    for pid in held:
        df = F[F.profile_id == pid]
        X = torch.from_numpy(df[input_cols].to_numpy(np.float32))[:, None, :]
        Y = df[target_cols].to_numpy(np.float32)
        pred, _ = model(X, torch.from_numpy(Y[:1]))
        P = pred[:, 0].numpy()
        cont[str(pid)] = {"mse_per_target_K2": [float(v) for v in (((P - Y) * 200) ** 2).mean(0)],
                          "mse_all_K2": float((((P - Y) * 200) ** 2).mean()), "max_abs_pm_K": float(np.abs(P[:, 0] - Y[:, 0]).max() * 200), "n": int(len(Y))}
        n_all = len(Y)
        for r in events(n_all):
            n = min(n_all, r + HOR); ks = [int(tm * 60 / TS) for tm in TIMES if r + int(tm * 60 / TS) < n]
            Xs = X[r:n]
            inits = {"ambient": np.full(4, Xs[0, 0, idx_amb].item(), np.float32),
                     "winding": np.full(4, Y[r, 3], np.float32),
                     "true": Y[r].copy()}
            for name, th0 in inits.items():
                pr, _ = model(Xs, torch.from_numpy(th0[None]))
                pm_hat = pr[:, 0, 0].numpy() * 200
                rec = {"fold": args.fold if args.mode == "cv" else -1, "seed": args.seed, "unit": pid, "r": r,
                       "event": "start" if r == 0 else "reset", "scenario": "S1", "method": f"TNN_{name}"}
                for k, tm in zip(ks, TIMES): rec[f"err_{tm}"] = float(abs(pm_hat[k] - Y[r + k, 0] * 200))
                rows.append(rec)
pd.DataFrame(rows).to_csv(out_dir / f"episodes_{tag}.csv", index=False)
json.dump({"tag": tag, "config": "TNN paper Table 2 optimum", "n_params": n_params, "train_units": train,
           "held_units": held, "final_train_mse_K2": final_train, "epochs": n_epochs, "jit": jit,
           "input_cols": input_cols, "continuous": cont, "runtime_s": time.time() - t0, "python": sys.version,
           "torch": torch.__version__, "platform": platform.platform()}, open(out_dir / f"meta_{tag}.json", "w"), indent=1)
print(tag, "DONE", round(time.time() - t0), flush=True)
