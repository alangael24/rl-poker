"""
Texas Hold'em Limit Poker - Self-Play con PufferLib
Objetivo: 5M+ steps/segundo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib
import pufferlib.pytorch
from pufferlib import pufferl
import gymnasium

# Backend C
try:
    import poker_c
    C_BACKEND = True
    print(">>> BACKEND C DISPONIBLE <<<")
except ImportError:
    C_BACKEND = False
    print(">>> BACKEND C NO DISPONIBLE <<<")
    print(">>> Ejecuta: python setup.py build_ext --inplace <<<")


# ============================================================================
# PufferEnv para Poker
# ============================================================================

class PokerPufferEnv(pufferlib.PufferEnv):
    """
    PufferEnv nativo para Texas Hold'em.
    Cada "agente" es una posición en una mesa de poker.
    """

    def __init__(self, num_envs=65536, num_players=6, small_blind=0.5,
                 big_blind=1.0, starting_stack=100.0, buf=None, seed=0, **kwargs):

        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(poker_c.OBS_SIZE,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(poker_c.NUM_ACTIONS)
        self.num_agents = num_envs

        super().__init__(buf)

        self._c_env = poker_c.PokerBatchEnv(
            num_envs=num_envs,
            num_players=num_players,
            small_blind=small_blind,
            big_blind=big_blind,
            starting_stack=starting_stack,
            seed=seed,
        )

        self._terms_u8 = np.zeros(num_envs, dtype=np.uint8)
        self._truncs_u8 = np.zeros(num_envs, dtype=np.uint8)

        self._c_env.set_buffers(self.observations, self.rewards,
                                self._terms_u8, self._truncs_u8)

        # Stats
        self._hands_played = 0
        self._total_reward = 0.0

    @property
    def emulated(self):
        return None

    def reset(self, seed=None):
        self._c_env.reset()
        return self.observations, []

    def step(self, actions):
        self._c_env.step(actions)
        self.terminals[:] = self._terms_u8
        self.truncations[:] = self._truncs_u8

        # Track stats
        self._hands_played += self.terminals.sum()
        self._total_reward += self.rewards.sum()

        return self.observations, self.rewards, self.terminals, self.truncations, []

    def close(self):
        pass


# ============================================================================
# Policy Network
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)

    def forward(self, x):
        return F.relu(self.ln2(self.fc2(F.relu(self.ln1(self.fc1(x))))) + x)


class PokerPolicy(nn.Module):
    def __init__(self, env, hidden_size=256, num_blocks=2):
        super().__init__()
        obs_size = np.prod(env.single_observation_space.shape)
        act_size = env.single_action_space.n

        self.input_fc = pufferlib.pytorch.layer_init(nn.Linear(obs_size, hidden_size))
        self.input_ln = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_blocks)])
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, act_size), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, x, state=None):
        x = x.float().view(x.shape[0], -1)
        x = F.relu(self.input_ln(self.input_fc(x)))
        for block in self.blocks:
            x = block(x)
        return self.actor(x), self.critic(x)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if not C_BACKEND:
        print("ERROR: Backend C no disponible")
        exit(1)

    print("\n" + "=" * 60)
    print("TEXAS HOLD'EM LIMIT POKER - SELF-PLAY")
    print("=" * 60)

    args = pufferl.load_config('default')
    args['train']['env'] = 'poker'
    args['train']['total_timesteps'] = 1_000_000_000  # 1B pasos
    args['train']['learning_rate'] = 3e-4
    args['train']['minibatch_size'] = 65536
    args['train']['bptt_horizon'] = 8
    args['train']['update_epochs'] = 1

    NUM_ENVS = 131072  # 128K envs para max throughput
    NUM_PLAYERS = 6

    vecenv = pufferlib.vector.make(
        PokerPufferEnv,
        env_kwargs={
            'num_envs': NUM_ENVS,
            'num_players': NUM_PLAYERS,
            'small_blind': 0.5,
            'big_blind': 1.0,
            'starting_stack': 100.0,
        },
        num_envs=1,
        backend=pufferlib.PufferEnv,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = PokerPolicy(vecenv, hidden_size=256, num_blocks=2).to(device)

    print(f"\n  Params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Envs: {NUM_ENVS:,}")
    print(f"  Players/table: {NUM_PLAYERS}")
    print(f"  OBS_SIZE: {poker_c.OBS_SIZE}")
    print(f"  Actions: Fold(0), Call(1), Raise(2)")
    print("=" * 60)

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    try:
        while trainer.epoch < trainer.total_epochs:
            trainer.evaluate()
            trainer.train()

            if trainer.epoch % 100 == 0:
                env = vecenv.envs[0] if hasattr(vecenv, 'envs') else vecenv
                avg_reward = env._total_reward / max(1, env._hands_played) if hasattr(env, '_hands_played') else 0
                print(f"  [Epoch {trainer.epoch}] Hands: {env._hands_played:,}, Avg reward: {avg_reward:.4f}")

    except KeyboardInterrupt:
        print("\nInterrumpido")

    trainer.print_dashboard()
    torch.save({'policy_state_dict': policy.state_dict()}, 'poker_model.pt')
    print("\nModelo guardado en poker_model.pt")
    trainer.close()
