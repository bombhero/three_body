import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import random
import numpy as np
import math


g_val = 6.67e-11


class FixedStar:
    def __init__(self, w_speed=1.0, w_position=1.0, single_point=None):
        self.position = np.array([[random.uniform(-7.4e10, 7.4e10), random.uniform(-7.4e10, 7.4e10)]]) * w_position
        self.speed = np.array([random.uniform(-29783, 29783), random.uniform(-29783, 29783)]) * w_speed
        self.acc = np.array([0, 0])
        self.radius = random.uniform(1, 6.95e8)
        self.weight = self.calc_weight(self.radius)
        self.existed = True
        if single_point is None:
            self.relative_position = np.array([[0, 0]])
        else:
            self.relative_position = self.position - single_point[(single_point.shape[0]-1):single_point.shape[0], :]

    @staticmethod
    def calc_weight(radius):
        return 6.3e3 * math.pi * math.pow(radius, 3) * 4 / 3

    @staticmethod
    def calc_radius(weight):
        return math.pow((weight * 4 / 6.3e3 / 4 / math.pi), 1/3)

    def next_step(self, power, single_point=None, time_inv=0.2):
        self.acc = power / self.weight
        if power[0] == 0 and power[1] == 0:
            self.speed = np.array([0.0, 0.0])
        self.speed += self.acc * time_inv
        x = self.position[-1, 0] + self.speed[0] * time_inv
        y = self.position[-1, 1] + self.speed[1] * time_inv
        new_position = np.array([[x, y]])
        self.position = np.concatenate((self.position, new_position), axis=0)

        if single_point is None:
            new_relative_position = np.array([[0, 0]])
        else:
            new_relative_position = np.array([self.position[-1, :] - single_point])
        self.relative_position = np.concatenate((self.relative_position, new_relative_position), axis=0)

    def get_position(self, n=100):
        if self.position.shape[0] < n:
            return self.position
        else:
            return self.position[-n:, :]

    def get_relative_position(self, n=100):
        if self.relative_position.shape[0] < n:
            return self.relative_position
        else:
            return self.relative_position[-n:, :]


class StarGroup:
    def __init__(self, n):
        self.group = []
        self.singular = -1
        self.group.append(FixedStar(w_position=0.1, w_speed=1))
        self.group[0].radius = 6.9e8
        self.group[0].weight = self.group[0].calc_weight(self.group[0].radius)
        for idx in range(1, n):
            self.group.append(FixedStar(single_point=self.group[0].get_position(), w_position=0.1, w_speed=1))

    def __len__(self):
        return len(self.group)

    def merge_star(self, major_idx, minor_idx):
        if not (self.group[major_idx].existed & self.group[minor_idx].existed):
            return
        if major_idx > minor_idx:
            tmp_idx = minor_idx
            minor_idx = major_idx
            major_idx = tmp_idx

        major_moment = self.group[major_idx].weight * self.group[major_idx].speed
        minor_moment = self.group[minor_idx].weight * self.group[minor_idx].speed
        final_moment = major_moment + minor_moment
        self.group[major_idx].weight = self.group[major_idx].weight
        self.group[major_idx].radius = self.group[major_idx].calc_radius(self.group[major_idx].weight)
        final_speed = final_moment / self.group[major_idx].weight
        self.group[major_idx].speed = np.array([final_speed[0], final_speed[1]])
        self.group[minor_idx].existed = False

    def live_star_count(self):
        count = 0
        for star in self.group:
            if star.existed:
                count += 1
        return count

    @staticmethod
    def univeral_gravitation(major_star, minor_star):
        x_diff = major_star.get_position()[-1, 0] - minor_star.get_position()[-1, 0]
        y_diff = major_star.get_position()[-1, 1] - minor_star.get_position()[-1, 1]
        length = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
        power_val = g_val * major_star.weight * minor_star.weight / math.pow(length, 2)
        power_x = 0 - power_val * x_diff / length
        power_y = 0 - power_val * y_diff / length
        return np.array([power_x, power_y]), length

    def next_step(self, time_inv=0.2):
        power = np.zeros([len(self), 2])
        for star_idx in range(len(self)):
            if not self.group[star_idx].existed:
                continue
            for idx in range(len(self)):
                if idx == star_idx:
                    continue
                if not self.group[idx].existed:
                    continue
                sub_power, length = self.univeral_gravitation(self.group[star_idx], self.group[idx])
                if length < self.group[star_idx].radius + self.group[idx].radius:
                    self.merge_star(star_idx, idx)
                    continue
                power[star_idx, :] += sub_power

        for star_idx in range(len(self)):
            if not self.group[star_idx].existed:
                continue
            if star_idx == 0:
                self.group[star_idx].next_step(power[star_idx, :], None, time_inv=time_inv)
            else:
                self.group[star_idx].next_step(power[star_idx, :], self.group[0].position[-1, :], time_inv=time_inv)


def main():
    plt.ion()
    fig = plt.figure(figsize=(18, 8))
    ax_total = fig.add_subplot(1, 2, 1)
    ax_total.axis('equal')
    ax_relative = fig.add_subplot(1, 2, 2)
    ax_relative.axis('equal')

    star_group = StarGroup(10)
    while star_group.live_star_count() > 1:
        star_group.next_step(time_inv=60)
        ax_total.cla()
        ax_relative.cla()
        for idx in range(len(star_group)):
            if not star_group.group[idx].existed:
                continue
            position = star_group.group[idx].get_position()
            ax_total.plot(star_group.group[idx].get_position()[:, 0],
                          star_group.group[idx].get_position()[:, 1])
            circle = mpatch.Circle(star_group.group[idx].get_position()[-1, :], star_group.group[idx].radius)
            ax_total.add_patch(circle)

        for idx in range(len(star_group)):
            if not star_group.group[idx].existed:
                continue
            ax_relative.plot(star_group.group[idx].get_relative_position(n=100000)[:, 0],
                             star_group.group[idx].get_relative_position(n=100000)[:, 1])
            circle = mpatch.Circle(star_group.group[idx].get_relative_position()[-1, :], star_group.group[idx].radius)
            ax_relative.add_patch(circle)
        plt.show()
        plt.pause(0.000001)
    plt.pause(2)


if __name__ == '__main__':
    main()
