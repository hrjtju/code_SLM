import torch
from torch import nn
from torch import Tensor
import pydantic

class PPO_Log(pydantic.BaseModel):
    avg_actor_loss: float
    avg_critic_loss: float
    avg_actor_grad_norm: float
    avg_critic_grad_norm: float

class DQN_Log(pydantic.BaseModel):
    loss: float
    grad_norm: float
    q_mean: float = 0.0
    td_error: float = 0.0


class RunningMeanStd:
    """
    T8: per-feature running mean/std used to normalise the observation.
    
    The raw SLM state mixes remaining areas (~1e5), volumes (~1e4), heights (~1e2)
    and part counts (~1e0) in a single flat vector. Feeding that into a Linear layer
    without normalisation gives wildly different effective learning rates per
    feature. Stats are frozen after a warm-up so the targets stop moving.
    """
    
    def __init__(self, dim: int, device: torch.device = "cpu", epsilon: float = 1e-4, clip: float = 10.0) -> None:
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = epsilon
        self.clip = clip
        self.frozen = False
    
    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        if self.frozen:
            return
        
        x = x.detach().float().reshape(-1, self.mean.shape[-1]).to(self.mean.device)
        batch_count = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        
        delta = batch_mean - self.mean
        total = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        
        self.mean, self.var, self.count = new_mean, m2 / total, total
    
    def normalize(self, x: Tensor) -> Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)
    
    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count, "frozen": self.frozen}
    
    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"].to(self.mean.device)
        self.var = d["var"].to(self.var.device)
        self.count = d["count"]
        self.frozen = d.get("frozen", True)

def calculate_gradient_norm(model: nn.Module) -> float:
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def compute_advantage(gamma: float, lmbda: float, td_delta: Tensor) -> Tensor:
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def init_weights_normal(m: torch.nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

def init_weights_ortho(m: torch.nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)