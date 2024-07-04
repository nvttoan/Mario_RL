import pickle
import random
from collections import deque

import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import matplotlib.pyplot as plt
from wrappers import *

# Chuyển đổi hình ảnh sang dữ liệu cần cho nơ ron
def arrange(s):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)

# Bộ nhớ để lưu trữ
class replay_memory(object):
    def __init__(self, N):
        self.memory = deque(maxlen=N)
    #Thêm một trải nghiệm vào bộ nhớ
    def push(self, transition):
        self.memory.append(transition)
    #Laays ngẫu nhiên từ bộ nhớ
    def sample(self, n):
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)

# Mô hình nơ ron
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
    # Forwar dự đoán Q
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - 1 / adv.shape[-1] * adv.max(-1, True)[0])

        return q


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train(q, q_target, memory, batch_size, gamma, optimizer, device):
    s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    s = np.array(s).squeeze()
    s_prime = np.array(s_prime).squeeze()
    a_max = q(s_prime).max(1)[1].unsqueeze(-1)
    r = torch.FloatTensor(r).unsqueeze(-1).to(device)
    done = torch.FloatTensor(done).unsqueeze(-1).to(device)
    with torch.no_grad():
        y = r + gamma * q_target(s_prime).gather(1, a_max) * done
    a = torch.tensor(a).unsqueeze(-1).to(device)
    q_value = torch.gather(q(s), dim=1, index=a.view(-1, 1).long())

    loss = F.smooth_l1_loss(q_value, y).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# Sao chép trọng số từ mô hình q sang mô hình q_target
def copy_weights(q, q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)

# Hàm chính để huấn luyện mô hình và chơi game
def main(env, q, q_target, optimizer, device):
    t = 0
    gamma = 0.99
    batch_size = 256

    N = 50000
    eps = 0.001
    memory = replay_memory(N)
    update_interval = 50
    print_interval = 10
    loss_lst = []
    score_lst = []
    total_score = 0.0
    loss = 0.0

    stuck_steps = 5  # Number of steps to detect if Mario is stuck
    stuck_threshold = 0.01  # Threshold to consider Mario stuck
    stuck_action = 5  # Action to perform when stuck (right + jump)
    prev_position = 0
    stuck_counter = 0

    try:
        for k in range(1000000):
            s = arrange(env.reset())
            done = False
            while not done:
                if eps > np.random.rand():
                    a = env.action_space.sample()
                else:
                    if device == "cpu":
                        a = np.argmax(q(s).detach().numpy())
                    else:
                        a = np.argmax(q(s).cpu().detach().numpy())

                s_prime, r, done, info = env.step(a)
                # print("Reward for this step:", r)
                print("action", a)
                s_prime = arrange(s_prime)
                total_score += r

                r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
                memory.push((s, float(r), int(a), s_prime, int(1 - done)))
                s = s_prime

                # Check if Mario is stuck
                current_position = info['x_pos']
                if abs(current_position - prev_position) < stuck_threshold:
                    stuck_counter += 1
                else:
                    stuck_counter = 0

                if stuck_counter >= stuck_steps:
                    a = stuck_action
                    stuck_counter = 0  # Reset the counter after taking the unstuck action

                prev_position = current_position

                if len(memory) > 2000:
                    loss += train(q, q_target, memory, batch_size, gamma, optimizer, device)
                    t += 1

                if t % update_interval == 0:
                    copy_weights(q, q_target)
                    torch.save(q.state_dict(), "mario_q.pth")
                    torch.save(q_target.state_dict(), "mario_q_target.pth")

            if k % print_interval == 0:
                avg_score = total_score / print_interval
                avg_loss = loss / print_interval
                print(
                    "%s |Epoch : %d | score : %f | loss : %.2f | stage : %d"
                    % (
                        device,
                        k,
                        avg_score,
                        avg_loss,
                        env.unwrapped._stage,
                    )
                )
                score_lst.append((k, avg_score))
                loss_lst.append((k, avg_loss))
                total_score = 0
                loss = 0.0
                pickle.dump(score_lst, open("score.p", "wb"))
                pickle.dump(loss_lst, open("loss.p", "wb"))

    finally:
        # Vẽ biểu đồ score
        if score_lst:
            epochs, scores = zip(*score_lst)
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, scores, label='Score')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Score vs Epoch')
            plt.legend()
            plt.savefig('score_vs_epoch.png')
            plt.close()

        # Vẽ biểu đồ loss
        if loss_lst:
            epochs, losses = zip(*loss_lst)
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, losses, label='Loss', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss vs Epoch')
            plt.legend()
            plt.savefig('loss_vs_epoch.png')
            plt.close()

if __name__ == "__main__":
    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
    env = JoypadSpace(env, RIGHT_ONLY)
    print(env.action_space)  # In ra không gian hành động
    print(RIGHT_ONLY)
    env = wrap_mario(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = model(n_frame, env.action_space.n, device).to(device)
    q_target = model(n_frame, env.action_space.n, device).to(device)
    optimizer = optim.Adam(q.parameters(), lr=0.0001)
    print(device)

    main(env, q, q_target, optimizer, device)
