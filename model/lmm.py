import torch
import torch.nn as nn
import torch.nn.functional as F

class LMM(nn.Module):
    def __init__(self, dim=768, ttt_lr=0.001, momentum=0.9, weight_decay=0.01, surprise_threshold=0.1):
        super().__init__()
        self.dim = dim
        self.ttt_lr = ttt_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.surprise_threshold = surprise_threshold

        self.net = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

        self.momentum_buffer = None
        self.ttt_optimizer = None

    def _get_optimizer(self):
        if self.ttt_optimizer is None:
            self.ttt_optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.ttt_lr,
                momentum=0.0,
                weight_decay=0.0
            )
        return self.ttt_optimizer

    def _apply_weight_decay(self):
        with torch.no_grad():
            for param in self.net.parameters():
                param.mul_(1.0 - self.weight_decay)

    def _update_momentum(self, surprise_magnitude):
        if self.momentum_buffer is None:
            self.momentum_buffer = surprise_magnitude
        else:
            self.momentum_buffer = (
                self.momentum * self.momentum_buffer +
                (1.0 - self.momentum) * surprise_magnitude
            )
        return self.momentum_buffer

    def forward(self, x, do_ttt=False):
        if do_ttt:
            self.net.train()
            opt = self._get_optimizer()

            for i, token in enumerate(x):
                # TTT every 4th token — 75% faster, minimal quality loss
                if i % 4 != 0:
                    continue

                token = token.unsqueeze(0).detach()

                # step 1 — surprise via gradient norm
                pred = self.net(token)
                memory_loss = F.mse_loss(pred, token)

                grads = torch.autograd.grad(
                    memory_loss,
                    self.net.parameters(),
                    create_graph=False,
                    allow_unused=True
                )

                surprise_magnitude = sum(
                    g.norm().item() for g in grads if g is not None
                )

                # step 2 — momentum
                effective_surprise = self._update_momentum(surprise_magnitude)

                # step 3 — threshold check
                if effective_surprise > self.surprise_threshold:

                    # step 4 — forgetting gate
                    self._apply_weight_decay()

                    # step 5 — recompute fresh after weight decay
                    pred_fresh = self.net(token)
                    loss_fresh = F.mse_loss(pred_fresh, token)

                    # step 6 — update weights
                    opt.zero_grad()
                    loss_fresh.backward()
                    opt.step()

            self.net.eval()

        with torch.no_grad():
            out = self.net(x)

        memory_summary = out.mean(dim=0, keepdim=True)
        return memory_summary

    def reset_memory_state(self):
        self.momentum_buffer = None

    def reset_ttt(self):
        self.momentum_buffer = None
        self.ttt_optimizer = None