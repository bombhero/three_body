import matplotlib.pyplot as plt
import random
import numpy as np
import math


g_val = 6.67


class FixedStar:
    def __init__(self, w_large=1.0, w_speed=1.0):
        self.position = np.array([[random.random(), random.random()]])
        self.speed = np.array([random.random(), random.random()]) * w_speed
        # self.speed = np.array([0.0, 0.0])
        self.acc = np.array([0, 0])
        self.weight = random.random() * w_large

    def next_step(self, power, time_inv=0.2):
        self.acc = power / self.weight
        if power[0] == 0 and power[1] == 0:
            self.speed = np.array([0.0, 0.0])
        self.speed += self.acc * time_inv
        x = self.position[-1, 0] + self.speed[0]
        y = self.position[-1, 1] + self.speed[1]
        new_position = np.array([[x, y]])
        self.position = np.concatenate((self.position, new_position), axis=0)

    def get_x(self):
        if self.position.shape[0] < 50:
            return self.position[:, 0]
        else:
            return self.position[-50:, 0]

    def get_y(self):
        if self.position.shape[0] < 50:
            return self.position[:, 1]
        else:
            return self.position[-50:, 1]


class StarGroup:
    def __init__(self, n):
        self.group = []
        for idx in range(n):
            self.group.append(FixedStar())

    def __len__(self):
        return len(self.group)


def univeral_gravitation(major_star, minor_star):
    x_diff = major_star.get_x()[-1] - minor_star.get_x()[-1]
    y_diff = major_star.get_y()[-1] - minor_star.get_y()[-1]
    length = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
    power_val = g_val * major_star.weight * minor_star.weight / math.pow(length, 2)
    power_x = 0 - power_val * x_diff / length
    power_y = 0 - power_val * y_diff / length
    if length < 0.1:
        power_x = 0.0
        power_y = 0.0
    return np.array([power_x, power_y])


def main():
    plt.ion()
    plt.show()

    fixed_star1 = FixedStar(w_large=100000, w_speed=0)
    fixed_star2 = FixedStar(w_large=1, w_speed=0.1)
    fixed_star2.position = fixed_star2.position * 10
    minor_line = fixed_star2.position - fixed_star1.position
    for idx in range(10000):
        plt.cla()
        star1_power = univeral_gravitation(fixed_star1, fixed_star2)
        star2_power = univeral_gravitation(fixed_star2, fixed_star1)
        fixed_star1.next_step(power=star1_power, time_inv=0.00001)
        fixed_star2.next_step(power=star2_power, time_inv=0.00001)
        minor_line = np.concatenate((minor_line, (fixed_star2.position[-2:-1, :] - fixed_star1.position[-2:-1, :])),
                                    axis=0)
        plt.plot(minor_line[:, 0], minor_line[:, 1], color='green')
        plt.scatter(0, 0, color='red')
        plt.scatter(minor_line[-1, 0], minor_line[-1, 1], s=fixed_star2.weight, color='green')
        # plt.plot(fixed_star1.get_x(), fixed_star1.get_y(), color='red')
        # plt.plot(fixed_star2.get_x(), fixed_star2.get_y(), color='green')
        # plt.scatter(fixed_star1.get_x()[-1], fixed_star1.get_y()[-1], color='red')
        # plt.scatter(fixed_star2.get_x()[-1], fixed_star2.get_y()[-1], color='green')
        if sum(fixed_star1.speed) == 0 or sum(fixed_star2.speed) == 0:
            break
        plt.pause(0.0001)
    plt.pause(2)


if __name__ == '__main__':
    main()
