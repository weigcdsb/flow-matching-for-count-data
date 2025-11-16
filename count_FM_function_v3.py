import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

#### FM for count 

def sample_xt(x0, x1, t):
    with torch.no_grad():
        diff = x1 - x0                     # [B,d]
        sign = torch.sign(diff)
        n    = diff.abs()                  # [B,d], integer counts
        t_full = t.expand_as(n)
        b = torch.binomial(n.float(), t_full)   # [B,d]
        return x0 + sign * b.to(x0.dtype)


def sample_rt(xt, x1, t, eps_t):
    # **un-normalized**
    # birth: lambda_t = |x1 - xt|/(1-t)
    # death: mu_t = |xt - x1|/(1-t)
    with torch.no_grad():
        idx_b = (x1 > xt).to(torch.float32) # birth id
        idx_d = (xt > x1).to(torch.float32) # death id
        idx_0 = (xt <= 0).to(torch.float32) # boundary, no death

        lambda_star = idx_b * (x1 - xt) / (1.0 - t + eps_t)        
        mu_star = idx_d * (xt - x1) / (1.0 - t + eps_t)*(1 - idx_0)
        
        # concatenate: [\lambda_{1:d}, \mu_{1:d}]
        rates_star = torch.cat([lambda_star, mu_star], dim=1)
        
        return rates_star, idx_0

def model_forward(xt_t, separate_heads, nets, d, idx_0):
    if not separate_heads:
        out = nets(xt_t)
        lambda_theta, beta_theta = out[:, :d], out[:, d:]
    else:
        net_b = nets[0]
        net_d = nets[1]
        lambda_theta = net_b(xt_t)
        beta_theta = net_d(xt_t)
        
    mu_theta = (xt_t[:, :d] * beta_theta) * (1.0 - idx_0)
    rates_theta = torch.cat([lambda_theta, mu_theta], dim=1)
    return rates_theta


def model_loss(loss_mode, rates_theta, rates_star, eps_log = 1e-8):
    if loss_mode == "l2":
        loss = ((rates_theta - rates_star)**2).sum(dim=1).mean()
    else:
        u = rates_star
        v = rates_theta
        loss = (v - u*torch.log(v + eps_log)).sum(1).mean()
        # loss = (u * (torch.log(u + eps_log) - torch.log(v + eps_log)) + v - u).sum(1).mean()
    
    return loss

def CountFM_train(
    X1_torch,
    nets,
    optimizer,
    num_epochs,
    batch_size,
    device,
    separate_heads=False,
    X0_torch=None,
    x0_mode="uniform",  # "uniform", "poisson", or "dataset"
    C_max=None,
    margin=2,
    eps_t=1e-4,
    eps_log=1e-8,
    loss_mode="poisson",
):
    X1_torch = X1_torch.to(device)
    N1, d = X1_torch.shape

    if X0_torch is not None:
        X0_torch = X0_torch.to(device)
        N0 = X0_torch.shape[0]
    else:
        N0 = N1

    if C_max is None:
        C_max = int(X1_torch.max().item() + margin)

    steps_per_epoch = (max(N0, N1) + batch_size - 1) // batch_size

    for epoch in tqdm(range(num_epochs)):
        perm1 = torch.randperm(N1, device=device)
        if X0_torch is not None and x0_mode == "dataset":
            perm0 = torch.randperm(N0, device=device)
        else:
            perm0 = None

        for step in range(steps_per_epoch):
            start = step * batch_size
            base  = torch.arange(batch_size, device=device)

            # sample x1 from data (like DFM)
            idx1 = perm1[(start + base) % N1]
            x1   = X1_torch[idx1]   # [B,d]
            B    = x1.shape[0]

            # construct x0 depending on mode
            if X0_torch is not None and x0_mode == "dataset":
                idx0 = perm0[(start + base) % N0]
                x0   = X0_torch[idx0]
            elif x0_mode == "uniform":
                # uniform prior over counts 0..C_max (D-FM-analogous)
                x0 = torch.randint(
                    low=0,
                    high=C_max + 1,
                    size=x1.shape,
                    device=device,
                    dtype=x1.dtype,
                )
            elif x0_mode == "poisson":
                lam0 = 1.0
                x0 = torch.poisson(
                    torch.full_like(x1, lam0, dtype=torch.float32)
                ).to(x1.dtype)
            else:
                raise ValueError(f"Unknown x0_mode: {x0_mode}")

            # 2. sample time U[eps_t, 1-eps_t]
            t = torch.rand(B, 1, device=device) * (1.0 - 2.0 * eps_t) + eps_t

            # 3. Binomial bridge
            xt = sample_xt(x0, x1, t)

            # 4. conditional rates
            rates_star, idx_0 = sample_rt(xt, x1, t, eps_t)

            # 5. model forward
            xt_t = torch.cat([xt, t], dim=1)   # [B, d+1]
            rates_theta = model_forward(xt_t, separate_heads, nets, d, idx_0)

            # 6. loss
            loss = model_loss(loss_mode, rates_theta, rates_star, eps_log)

            # 7. backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if epoch % max(1, (num_epochs // 5)) == 0:
            print(f"[CountFM][epoch {epoch}] loss={float(loss):.6f}")

    return nets, loss

#### genearation

def sample_euler(nets, n_step, x0, device, eps_t = 1e-4, eps_log = 1e-8, separate_heads=False):
    xt = x0.to(device=device, dtype=torch.float32).clone()
    N, d = xt.shape
    t = torch.full((N, 1), eps_t, device=device, dtype=torch.float32)
    Delta = torch.tensor((1.0 - 2.0*eps_t) / float(n_step),
                         device=device, dtype=torch.float32)
    traj = torch.empty(n_step + 1, N, d, device=device, dtype=torch.float32)
    traj[0] = xt

    for s in range(n_step):
        xt_t = torch.cat([xt, t], dim=1)
        if not separate_heads:
            out = nets(xt_t)  # [N, 2d]
            lambda_theta, beta_theta = out[:, :d], out[:, d:]
        else:
            lambda_theta = nets[0](xt_t)  # [N, d]
            beta_theta = nets[1](xt_t)  # [N, d]

        idx_0 = (xt <= 0).to(torch.float32)         # no death at 0
        mu_theta = (xt * beta_theta) * (1.0 - idx_0)

        r_i = (lambda_theta + mu_theta)             # [N, d]
        p_none = torch.exp(-r_i * Delta)            # stay
        p_jump = 1.0 - p_none                       # jump

        p_birth = p_jump * (lambda_theta / (r_i + eps_log))
        p_death = p_jump * (mu_theta     / (r_i + eps_log))
        probs3 = torch.stack([p_none, p_birth, p_death], dim=-1)  # [N,d,3]

        probs3_flat = probs3.reshape(-1, 3)
        choice = torch.multinomial(probs3_flat, 1).view(N, d)

        adj = (choice == 1).to(torch.float32) - (choice == 2).to(torch.float32)
        xt = torch.clamp(xt + adj, min=0.0)

        t = torch.minimum(t + Delta, torch.full_like(t, 1.0 - eps_t))
        traj[s + 1] = xt

    x1_samples = xt.to(torch.long)
    return x1_samples, traj

        
##### evaluation

def basic_stats(x):
    mean = x.mean(0)
    var = x.var(0, unbiased=False)
    zero_frac = (x == 0).float().mean(0)
    return mean, var, zero_frac

def corr_mat(x, eps=1e-8):
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / x.shape[0]
    std = cov.diag().clamp_min(eps).sqrt()
    return cov / (std[:, None] * std[None, :])


def mmd_rbf(x, y, gamma=None):
    # x:[n,d], y:[m,d]
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    def _kernel(a, b):
        # ||a-b||^2
        a2 = (a*a).sum(1, keepdim=True)
        b2 = (b*b).sum(1, keepdim=True)
        dist2 = a2 - 2.0 * (a @ b.T) + b2.T
        return torch.exp(-gamma * dist2)

    Kxx = _kernel(x, x)
    Kyy = _kernel(y, y)
    Kxy = _kernel(x, y)

    # use unbiased-ish version
    n = x.shape[0]
    m = y.shape[0]
    mmd2 = (Kxx.sum() - Kxx.diag().sum()) / (n*(n-1) + 1e-8) \
         + (Kyy.sum() - Kyy.diag().sum()) / (m*(m-1) + 1e-8) \
         - 2.0 * Kxy.mean()
    return mmd2.item()



        

