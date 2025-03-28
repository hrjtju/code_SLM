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

def init_weights_normal(m: torch.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

def init_weights_ortho(m: torch.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)