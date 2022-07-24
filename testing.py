import marlgrid.envs
import gym
import warnings
import random
import cv2

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

if __name__ == '__main__':
    env = gym.make("CoordinationGoaltile-5Agents-10Goals-2coordination-v1")
    obs = env.reset()

    image = env.grid.render(tile_size=8)

    cv2.imwrite('Coordination2.png', image)

    actions = [[2, 2], [2, 2], [0, 1], [2, 2], [2, 2]]
    for i in range(5):
        action = actions[i]
        obs, reward, done, info = env.step(action)
        image = env.grid.render(tile_size=8)
        cv2.imwrite(f'Coordination2_{i}.png', image)
        print(f'reward: {reward}')
