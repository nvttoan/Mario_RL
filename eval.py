import sys
import time

import gym_super_mario_bros
import torch
import torch.nn as nn
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT,RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import gym
from gym.wrappers import RecordVideo
from wrappers import *
import os

# Mô hình mạng nơ-ron
class model(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(20736, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)

        self.device = device
        self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, self.q, self.v)

        self.seq.apply(init_weights)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        #Tính toán giá trị Q  sử dụng công thức dueling DQN.
        q = v + (adv - 1 / adv.shape[-1] * adv.max(-1, True)[0])
        return q


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Kiểm tra dữ liệu đầu vào
def arange(s):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1)) # chuyển thành channels, height, width cho mạng nơ ron
    return np.expand_dims(ret, 0) # thêm chiều đầu tiên( batch size)


if __name__ == "__main__":
    # Đường dẫn đến file checkpoint
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "mario_q_target.pth"
    print(f"Load ckpt from {ckpt_path}")
    # Số frame liên tiếp sử dụng để dự đoán hành động
    n_frame = 4
    # Tạo môi trường Super Mario Bros
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, RIGHT_ONLY)
    env = wrap_mario(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tải trọng số mạng từ checkpoint
    q = model(n_frame, env.action_space.n, device).to(device)

    q.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))

    # video_dir = './videos'
    # os.makedirs(video_dir, exist_ok=True)
    while True:
        total_score = 0.0
        done = False
        s = arange(env.reset())
        i = 0

        # # Start recording video
        # video_path = os.path.join(video_dir, f"run_{time.time()}.mp4")
        # env = gym.wrappers.Monitor(env, video_path, force=True)
        # Vào game
        while not done:
            env.render()
            # Dự đoán hành động tốt nhất dựa trên trạng thái hiện tại
            if device == "cpu":
                a = np.argmax(q(s).detach().numpy())
            else:
                a = np.argmax(q(s).cpu().detach().numpy())
            # Thực hiện hành động và nhận lại trạng thái mới và thưởng
            s_prime, r, done, _ = env.step(a)
            s_prime = arange(s_prime)
            total_score += r
            s = s_prime
            time.sleep(0.001)
        # Lấy thông tin về màn chơi và điểm số tổng cộng
        stage = env.unwrapped._stage
        print("Total score : %f | stage : %d" % (total_score, stage))

