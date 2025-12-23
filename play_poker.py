"""
Jugar Texas Hold'em contra el modelo entrenado.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pufferlib.pytorch

# Import poker_c
try:
    import poker_c
except ImportError:
    print("ERROR: Ejecuta primero: python setup.py build_ext --inplace")
    exit(1)


# ============================================================================
# Policy (igual que en train_poker.py)
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
    def __init__(self, obs_size=139, act_size=3, hidden_size=256, num_blocks=2):
        super().__init__()
        self.input_fc = nn.Linear(obs_size, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_blocks)])
        self.actor = nn.Linear(hidden_size, act_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, state=None):
        x = x.float().view(x.shape[0], -1)
        x = F.relu(self.input_ln(self.input_fc(x)))
        for block in self.blocks:
            x = block(x)
        return self.actor(x), self.critic(x)

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            logits, _ = self.forward(obs.unsqueeze(0))
            if deterministic:
                return logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1).item()


# ============================================================================
# Card Display
# ============================================================================

SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

def card_to_str(card_idx):
    rank = card_idx % 13
    suit = card_idx // 13
    return f"{RANKS[rank]}{SUITS[suit]}"


def print_cards(cards, label=""):
    if label:
        print(f"{label}: ", end="")
    print(" ".join(card_to_str(c) for c in cards))


# ============================================================================
# Simple Single-Player Game
# ============================================================================

class PokerGame:
    def __init__(self, policy, device='cpu'):
        self.policy = policy
        self.device = device
        self.env = poker_c.PokerBatchEnv(
            num_envs=1,
            num_players=2,  # Heads-up: Player vs AI
            small_blind=0.5,
            big_blind=1.0,
            starting_stack=100.0,
            seed=42,
        )
        self.player_seat = 0  # Human is seat 0
        self.ai_seat = 1      # AI is seat 1

    def get_observation(self, env_idx=0):
        obs, _ = self.env.reset()
        return obs[env_idx]

    def play_hand(self):
        obs, _ = self.env.reset()
        obs = obs[0]  # Single env

        hand_over = False
        actions_taken = []

        print("\n" + "=" * 50)
        print("NUEVA MANO")
        print("=" * 50)

        while not hand_over:
            # Decode observation to show cards
            # First 52 values are hole cards (one-hot)
            hole_cards = []
            for i in range(52):
                if obs[i] > 0.5:
                    hole_cards.append(i)

            # Next 52 values are community cards
            community_cards = []
            for i in range(52, 104):
                if obs[i] > 0.5:
                    community_cards.append(i - 52)

            print(f"\nTus cartas: ", end="")
            print_cards(hole_cards)
            if community_cards:
                print(f"Mesa: ", end="")
                print_cards(community_cards)

            # Get current player from observation context
            # For simplicity, alternate turns
            current_player = len(actions_taken) % 2

            if current_player == self.player_seat:
                # Human turn
                print("\nTu turno:")
                print("  [0] Fold")
                print("  [1] Call/Check")
                print("  [2] Raise")

                while True:
                    try:
                        action = int(input("Acción: "))
                        if action in [0, 1, 2]:
                            break
                        print("Acción inválida. Usa 0, 1, o 2.")
                    except ValueError:
                        print("Ingresa un número.")
            else:
                # AI turn
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = self.policy.get_action(obs_tensor, deterministic=True)
                action_names = ['Fold', 'Call', 'Raise']
                print(f"\nAI: {action_names[action]}")

            # Take action
            actions = np.array([action], dtype=np.int32)
            obs, rewards, terminals, truncations, info = self.env.step(actions)
            obs = obs[0]
            reward = rewards[0]
            hand_over = terminals[0] or truncations[0]
            actions_taken.append(action)

        # Hand over
        print("\n" + "-" * 50)
        if reward > 0:
            print(f"¡GANASTE! Reward: +{reward:.2f}")
        elif reward < 0:
            print(f"Perdiste. Reward: {reward:.2f}")
        else:
            print(f"Empate. Reward: {reward:.2f}")

        return reward


def main():
    print("\n" + "=" * 50)
    print("TEXAS HOLD'EM - HUMANO VS AI")
    print("=" * 50)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = PokerPolicy().to(device)

    try:
        checkpoint = torch.load('poker_model.pt', map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print("Modelo cargado: poker_model.pt")
    except FileNotFoundError:
        print("AVISO: No se encontró poker_model.pt, usando pesos aleatorios")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        print("Usando pesos aleatorios")

    policy.eval()

    game = PokerGame(policy, device)

    total_reward = 0
    hands_played = 0

    print("\nComandos: juega manos, escribe 'q' para salir")

    while True:
        try:
            reward = game.play_hand()
            total_reward += reward
            hands_played += 1
            print(f"\nBalance total: {total_reward:+.2f} ({hands_played} manos)")

            cont = input("\n¿Otra mano? (Enter=sí, q=salir): ")
            if cont.lower() == 'q':
                break
        except KeyboardInterrupt:
            break

    print(f"\n{'=' * 50}")
    print(f"RESUMEN FINAL")
    print(f"Manos jugadas: {hands_played}")
    print(f"Balance: {total_reward:+.2f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
