import numpy as np
import gym
import matplotlib.pyplot as plt


class QTable:

    def __init__(self, limits, n=50):
        cp, cv, pa, pv = limits

        self.limits = limits
        self.step = {}

        self.step['cart_position'] = (abs(cp[0]) + abs(cp[1])) / n
        self.step['cart_velocity'] = (abs(cv[0]) + abs(cv[1])) / n
        self.step['pole_angle']    = (abs(pa[0]) + abs(pa[1])) / n
        self.step['pole_velocity'] = (abs(pv[0]) + abs(pv[1])) / n

        self.states = np.random.normal(0, 1, size=(n**4,2))

        print("Creating QTable(limits={}, n={})... table shape={}".format(limits, n, self.states.shape))
                                         
    def approximate_state(self, obs):
        cp, cv, pa, pv = obs

        gap_cp = self.limits[0][0] - cp
        gap_cv = self.limits[1][0] - cv
        gap_pa = self.limits[2][0] - pa
        gap_pv = self.limits[3][0] - pv

        i_approx_cp = abs(gap_cp) / self.step['cart_position']
        i_approx_cv = abs(gap_cv) / self.step['cart_velocity']
        i_approx_pa = abs(gap_pa) / self.step['pole_angle']
        i_approx_pv = abs(gap_pv) / self.step['pole_velocity']

        state_index = i_approx_cp * i_approx_cv * i_approx_pa * i_approx_pv

        return int(round(state_index) - 1)


# FROM COOKBOOK
def play_one(env, qtable, eps, alpha, gamma, log=True):
    obs = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 2000:

        # LOG
        if log:
            print("Iter {}".format(iters))

        # Select action
        prev_state = qtable.approximate_state(obs)
        q_sa = qtable.states[prev_state]
        action = np.argmax(q_sa)

        # LOG
        if log:
            print("---> In state {} with Q(s, a) = {} selected action {}.".
                    format(prev_state, q_sa, action))

        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            reward = -400

        # Calculate gain G 
        next_state = qtable.approximate_state(obs)
        q_sa_prime = qtable.states[next_state]
        G = reward + gamma * np.max(q_sa_prime)

        # LOG
        if log:
            print("---> Gain of {} from state {} with Q(s', a') = {} and r = {}.".
                    format(G, next_state, q_sa_prime, reward))

        # Q(s, a) update
        prev_q_sa = qtable.states[prev_state]
        qtable.states[prev_state][action] = (1 - alpha) * prev_q_sa[action] + alpha * G
        actual_q_sa = qtable.states[prev_state]
        
        # LOG
        if log:
            print("---> Updating state {}: {} to {}".
                    format(prev_state, prev_q_sa, actual_q_sa))


        if reward == 1:
            totalreward += reward
        
        iters += 1

    return totalreward


if __name__ == '__main__':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    gamma = 0.90
    alpha = 0.01

    N = 10000
    totalrewards = np.empty(N)
    running_avg = np.empty(N)

    limits = [(-2.5, 2.5), (-3.5, 3.5), (-0.30, 0.30), (-4, 4)]
    n = 100
    qtable = QTable(limits, n)

    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        totalreward = play_one(env, qtable, eps, alpha, gamma, log=False)
        totalrewards[n] = totalreward
        running_avg[n] = totalrewards[max(0, n - 100):(n + 1)].mean()
        if n % 100 == 0:
            print("episode: {0}, total reward: {1}, eps: {2}, avg reward (last 100): {3}".format(
                n, totalreward, eps, running_avg[n]),
                )

    print("avg reward for last 100 episodes: ", totalrewards[-100:].mean())
    print("total steps: ", totalrewards.sum())

    plt.plot(totalrewards)
    plt.xlabel('episodes')
    plt.ylabel('Total Rewards')
    plt.show()

    plt.plot(running_avg)

    plt.xlabel('episodes')
    plt.ylabel('Running Average')
    plt.show()

    env.close()

