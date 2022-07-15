import gym
import multiprocessing as mp

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        # print(f'Received command: {cmd} and action {data}')
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        self.processes = []
        for env in self.envs[1:]:
            local, remote = mp.Pipe()
            self.locals.append(local)
            p = mp.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        # print("Resetting")
        i = 0
        for local in self.locals:
            # print(f'Sending the reset command to process {i}')
            local.send(("reset", None))
            i += 1
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        # print(f"Taking actions: {actions} of type {actions.dtype}")
        i = 0
        for local, action in zip(self.locals, actions[1:]):
            # print(f"Sending action {action} to process {i}")
            local.send(("step", action))
            i += 1
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()