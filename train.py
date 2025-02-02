import numpy as np
from tic_tac_toe import TicTacToe
from agent import DQNAgent
import random

def train_agent_self_play(p1, p2, env, num_episodes=50000):
    for episode in range(num_episodes):
        if episode % 500 == 0:
            p1.epsilon = max(p1.epsilon * 0.9, 0.1)
            p2.epsilon = max(p2.epsilon * 0.9, 0.1)
            p1.update_epsilon(p1.epsilon)
            p2.update_epsilon(p2.epsilon)
            p2.update_target_network()
            
        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes} | lr={p1.lr:.6f} | epsilon={p1.epsilon:.4f}")
            print(f"\n--- Testing Agent at Episode {episode} ---")
            test_agent(p1, env, n_games=100, opponent="random")
            print("-------------------------------\n")
        


        # Rest of the training logic remains the same
        env.reset()
        done = False

        while not done:
            # p1 turn
            positions = env.available_actions()
            state = env.get_state()
            action = p1.act(state, valid_actions=positions)
            next_state, reward, done = env.step(action)
            
            p1.remember(state, action, reward, next_state, done)
            
            winner = env.winner
            if winner is not None:
                reward_p1, reward_p2 = _get_rewards(winner, p1_symbol=1)
                p1.remember(state, action, reward_p1, next_state, done)
                p2.remember(state, action, reward_p2, next_state, done)
                p1.replay()
                p2.replay()
                break
            else:
                # p2 turn
                positions = env.available_actions()
                state = env.get_state()
                action = p2.act(state, valid_actions=positions)
                next_state, reward, done = env.step(action)
                
                p2.remember(state, action, reward, next_state, done)
                
                winner = env.winner
                if winner is not None:
                    reward_p1, reward_p2 = _get_rewards(winner, p1_symbol=1)
                    p1.remember(state, action, reward_p1, next_state, done)
                    p2.remember(state, action, reward_p2, next_state, done)
                    p1.replay()
                    p2.replay()
                    break

    # Final test after all episodes
    print("\n--- Final Test Results ---")
    test_agent(p1, env, n_games=1000, opponent="random")
    
    p1.save("policy_p1.pt")
    p2.save("policy_p2.pt")
    print("Training complete.")


def train_agent_vs_random(agent, env, num_episodes=50000):
    for episode in range(num_episodes):
        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes}")
            
        env.reset()
        done = False

        while not done:
            # Agent's turn
            positions = env.available_actions()  # Fixed
            state = env.get_state()
            action = agent.act(state, valid_actions=positions)
            next_state, reward, done = env.step(action)  # Fixed
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            if done:
                break

            # Random opponent's turn
            positions = env.available_actions()  # Fixed
            if positions:
                opp_action = random.choice(positions)
                env.step(opp_action)  # Fixed

    agent.save("policy_vs_random.pt")
    print("Training vs random complete.")

def test_agent(agent, env, n_games=1000, opponent="random"):
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    wins, losses, draws = 0, 0, 0

    for _ in range(n_games):
        env.reset()
        done = False

        while not done:
            # Agent's turn
            positions = env.available_actions()  # Fixed method name
            state = env.get_state()
            action = agent.act(state, valid_actions=positions)  # Use DQNAgent's act()
            next_state, reward, done = env.step(action)  # Fixed method name

            winner = env.winner
            if winner is not None:
                if winner == 1:
                    wins += 1
                elif winner == 0:
                    draws += 1
                else:
                    losses += 1
                break
            else:
                # Opponent's turn
                if opponent == "random":
                    positions = env.available_actions()  # Fixed
                    if positions:
                        opp_action = random.choice(positions)
                        env.step(opp_action)  # Fixed method name

                winner = env.winner
                if winner is not None:
                    if winner == -1:
                        losses += 1
                    elif winner == 0:
                        draws += 1
                    else:
                        wins += 1
                    break

    agent.epsilon = original_epsilon
    print(f"Results: Wins={wins}, Losses={losses}, Draws={draws}")
    return wins, losses, draws


def _get_rewards(winner, p1_symbol=1):
    """
    Helper to assign rewards. If p1_symbol=1, then:
      - if winner= 1 => p1 gets +1
      - if winner=-1 => p2 gets +1
      - if winner= 0 => a draw: partial reward
    """
    if winner == p1_symbol:
        return (1, -1)
    elif winner == 0:
        # e.g. p1 gets 0.1, p2 gets 0.5 -- or pick your own
        return (0.1, 0.1)
    else:
        # p2 must have won
        return (-1, 1)


if __name__ == "__main__":
    # Example usage:

    # 1) Self-play training
    env = TicTacToe()
    p1 = DQNAgent()
    p2 = DQNAgent()
    train_agent_self_play(p1, p2, env, num_episodes=10000)

    # Load an agent and test it vs random:
    test_agent(p1, env, n_games=1000, opponent="random")

    # 2) Train vs random
    # agent_vs_random = QLearningAgent(lr=0.2, gamma=0.9, epsilon=0.3)
    # train_agent_vs_random(agent_vs_random, env, num_episodes=10000)
    # test_agent(agent_vs_random, env, n_games=1000, opponent="random")
