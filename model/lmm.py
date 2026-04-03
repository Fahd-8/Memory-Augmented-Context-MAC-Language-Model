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

            for token in x:
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

        # compress entire sequence into single memory summary vector
        memory_summary = out.mean(dim=0, keepdim=True)  # shape [1, dim]
        return memory_summary

    def reset_memory_state(self):
        # call between sequences — only resets momentum
        self.momentum_buffer = None

    def reset_ttt(self):
        # call between epochs — full reset including optimizer
        self.momentum_buffer = None
        self.ttt_optimizer = None