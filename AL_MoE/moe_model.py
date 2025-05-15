import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Expert 模块
# -----------------------
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(last_dim, h), nn.ReLU()])
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# -----------------------
# Gating 网络
# -----------------------
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.fc(x)  # [B, E]
        weights = F.softmax(logits, dim=-1)
        return weights

# -----------------------
# MoE 主模型（YTS单目标）
# -----------------------
class MoE_YTS(nn.Module):
    def __init__(self, input_dim, expert_hidden, num_experts=8, expert_output_dim=32):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden, expert_output_dim) for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(input_dim, num_experts)
        self.output_head = nn.Linear(expert_output_dim, 1)

    def forward(self, x):
        gating_weights = self.gating(x)  # [B, E]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, E, D]
        gating_weights = gating_weights.unsqueeze(-1)  # [B, E, 1]
        fused = torch.sum(gating_weights * expert_outputs, dim=1)  # [B, D]
        yts_pred = self.output_head(fused).squeeze(-1)  # [B]
        return yts_pred
