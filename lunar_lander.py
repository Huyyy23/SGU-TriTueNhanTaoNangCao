import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def random_agent(observation):
    return np.random.choice([0, 1, 2, 3])


def simple_reflex_agent(observation):
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation

    if vy < -0.4:
        return 2

    return 0


def improved_reflex_agent(observation):
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation

    if angle > 0.20:
        return 3

    if angle < -0.20:
        return 1

    if angular_velocity > 0.30:
        return 1

    if angular_velocity < -0.30:
        return 3

    if x > 0.15:
        return 1

    if x < -0.15:
        return 3

    if vx > 0.20:
        return 1

    if vx < -0.20:
        return 3

    if vy < -0.40:
        return 2

    if y < 0.25 and vy < -0.20:
        return 2

    return 0


def sensor_model(observation):
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation

    distance_to_center = abs(x)

    if vy < -0.05:
        descent_time = y / abs(vy)
    else:
        descent_time = 2.0

    descent_time = np.clip(descent_time, 0.1, 5.0)

    predicted_x = x + vx * descent_time

    return {
        "distance_to_center": distance_to_center,
        "descent_time": descent_time,
        "predicted_x": predicted_x
    }


def model_based_sensor_agent(observation):
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation

    sensors = sensor_model(observation)

    distance_to_center = sensors["distance_to_center"]
    descent_time = sensors["descent_time"]
    predicted_x = sensors["predicted_x"]

    if angle > 0.20:
        return 3

    if angle < -0.20:
        return 1

    if angular_velocity > 0.30:
        return 1

    if angular_velocity < -0.30:
        return 3

    if predicted_x > 0.15:
        return 1

    if predicted_x < -0.15:
        return 3

    if abs(predicted_x) < 0.12:
        if abs(vx) > 0.20:
            if vx > 0:
                return 1
            else:
                return 3

    if vy < -0.40:
        return 2

    if y < 0.35 and vy < -0.20:
        return 2

    if distance_to_center < 0.10 and abs(vx) < 0.10:
        if vy < -0.15:
            return 2

    return 0


def run_episode(env, agent, max_steps=1000):
    observation, info = env.reset()

    total_reward = 0.0

    for step in range(max_steps):
        action = agent(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    return total_reward


def evaluate_agent(env, agent, episodes=100):
    rewards = []

    for episode in range(episodes):
        reward = run_episode(env, agent)
        rewards.append(reward)

    rewards = np.array(rewards)

    return {
        "rewards": rewards,
        "average": np.mean(rewards),
        "std": np.std(rewards),
        "min": np.min(rewards),
        "max": np.max(rewards),
        "success_rate": np.mean(rewards >= 200) * 100
    }


env = gym.make("LunarLander-v3")


agents = {
    "Random Agent": random_agent,
    "Simple Reflex": simple_reflex_agent,
    "Improved Reflex": improved_reflex_agent,
    "Model-Based + Sensor": model_based_sensor_agent
}


results = {}


for name, agent in agents.items():

    print("=" * 60)
    print(name)

    result = evaluate_agent(
        env,
        agent,
        episodes=100
    )

    results[name] = result

    print(f"Average reward : {result['average']:.2f}")
    print(f"Std reward     : {result['std']:.2f}")
    print(f"Minimum reward : {result['min']:.2f}")
    print(f"Maximum reward : {result['max']:.2f}")
    print(f"Success rate   : {result['success_rate']:.2f}%")


env.close()


names = list(results.keys())

average_rewards = [
    results[name]["average"]
    for name in names
]

std_rewards = [
    results[name]["std"]
    for name in names
]


plt.figure(figsize=(11, 6))

plt.bar(
    names,
    average_rewards,
    yerr=std_rewards,
    capsize=5
)

plt.axhline(
    y=200,
    linestyle="--",
    label="Success threshold"
)

plt.xlabel("Agent")
plt.ylabel("Average Episode Reward")
plt.title("Lunar Lander Agent Comparison")

plt.legend()
plt.tight_layout()
plt.show()


print()
print("=" * 85)
print("FINAL COMPARISON")
print("=" * 85)

print(
    f"{'Agent':<25}"
    f"{'Average':>12}"
    f"{'Std':>12}"
    f"{'Min':>12}"
    f"{'Max':>12}"
    f"{'Success':>12}"
)

print("-" * 85)

for name in names:

    result = results[name]

    print(
        f"{name:<25}"
        f"{result['average']:>12.2f}"
        f"{result['std']:>12.2f}"
        f"{result['min']:>12.2f}"
        f"{result['max']:>12.2f}"
        f"{result['success_rate']:>11.2f}%"
    )