import os

import numpy as np

os.environ.setdefault("PATH", "")
from collections import deque

import cv2
import gym
from gym import spaces

cv2.ocl.setUseOpenCL(False)
from gym.wrappers import TimeLimit

# Wrapper để thêm hoạt động "noop" (no operation) vào quá trình reset
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Các trạng thái ban đầu thực hiện một số ngẫu nhiên các hoạt động "noop" trong quá trình reset.
        giúp ngẫu nhiên hóa trạng thái khởi đầu của môi trường
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Thực hiện hoạt động no-op trong một số bước trong khoảng từ 1 đến noop_max."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

# Wrapper để thực hiện hoạt động "fire" (bắn) trong quá trình reset
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Thực hiện hoạt động trong quá trình reset cho các môi trường được cố định cho đến khi bắn."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

# Wrapper để xử lý kết thúc cuộc đời của nhân vật trong môi trường Mario
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Kết thúc một cuộc đời = kết thúc một tập, nhưng chỉ reset khi game thực sự kết thúc."""
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # Kiểm tra số lượng cuộc sống hiện tại, khi mất một cuộc sống, kết thúc tập,
        # sau đó cập nhật số lượng cuộc sống để xử lý thêm các cuộc sống bonus
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # Đối với trường hợp Qbert, đôi khi chúng ta ở trong tình trạng lives == 0 trong một vài frame
            # Vì vậy việc giữ cho lives > 0 là quan trọng, để chúng ta chỉ reset một lần
            # khi môi trường thông báo rằng tập đã kết thúc.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset chỉ khi cuộc sống đã cạn kiệt.
        Điều này giúp tất cả các trạng thái vẫn có thể đạt được mặc dù cuộc sống là episodic,
        và người học không cần biết về bất kỳ điều này ở phía sau cánh cửa.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # Bước không hoạt động để tiến từ trạng thái kết thúc/ mất cuộc sống
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

# Wrapper để chỉ trả về mỗi khung hình thứ k
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Chỉ trả về mỗi khung hình thứ `skip`"""
        gym.Wrapper.__init__(self, env)
        # các quan sát gốc mới nhất (để lấy max qua các bước thời gian)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Lặp lại hoạt động, tính tổng thưởng và max qua các quan sát gần đây."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Lưu ý rằng quan sát trong frame done=True không quan trọng
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info


    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Chia thưởng thành {+1, 0, -1} ."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Chuyển đổi khung hình thành 84x84.
         `dict_space_key` có thể được xác định quan sát nào sẽ được biến đổi.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

# Wrapper để xếp các khung hình
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Xếp k khung hình cuối cùng lại vơí nhau.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """Đối tượng này đảm bảo rằng các frame phổ biến giữa các quan sát chỉ được lưu một lần.
        Đây chỉ là một biện pháp để tối ưu hóa việc sử dụng bộ nhớ của DQN.
        Đối tượng này chỉ nên được chuyển thành mảng numpy trước khi được truyền cho mô hình."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind(
    env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True
):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class EpisodicLifeMario(gym.Wrapper):
    def __init__(self, env):
        """Kết thúc 1 episode khi thoi gian đã hết dù mario chưa chết
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset lại trò chơi khi Mariodđã chết
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs


def wrap_mario(env):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeMario(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    # env = custom_reward(env)
    env = FrameStack(env, 4)
    return env
