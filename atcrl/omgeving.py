import gym
import pygame
from gym.spaces import Box

from atcrl.utils import *


class ATCEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_aircraft=2):
        # simulation variables
        self.radius_nm = 30
        self.seconds_per_step = 3
        self.num_aircraft = num_aircraft
        self.turn_rate_radps = 5 * np.pi / 180
        self.alt_rate_fpm = 1000
        self.speed_rate_kps = 1
        self.localiser_range_nm = 25
        self.glideslope_fpm = 700
        self.final_approach_point_nm = 10

        self.dones = np.zeros(num_aircraft, dtype=bool)
        # populated in self.reset
        self.aircraft = np.zeros((8, num_aircraft))

        # scoring variables
        self.minimum_separation_distance_nm = 3
        self.minimum_separation_distance_reward = -1
        self.left_airspace_reward = -1
        self.passive_time_reward = 0
        self.altitude_reward_per_100feet_off = -0.01
        self.heading_off_reward = -0.5
        self.established_reward = 100

        # rendering variables
        self.window_size = 512
        self.miles_per_pixel = 2 * self.radius_nm / self.window_size
        self.fps = 4
        self.header_line_length_nm = 0.5
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.font = None
        self.clock = None

        self.observation_space = Box(-1, 1, shape=(5 * num_aircraft,))
        self.action_space = Box(-1, 1, shape=(3 * num_aircraft,))

    def reset(self, *, seed=None):
        super().reset(seed=seed)  # setup RNG correctly

        # seed starting angle
        spawn_angle = self.np_random.random(self.num_aircraft) * 2 * np.pi
        # heading the opposite direction
        heading = normalize_angle(spawn_angle - np.pi)

        self.aircraft[0] = self.radius_nm * np.cos(spawn_angle)
        self.aircraft[1] = self.radius_nm * np.sin(spawn_angle)
        self.aircraft[2] = 7_000
        self.aircraft[3] = heading
        self.aircraft[4] = 250
        self.aircraft[5] = 7_000
        self.aircraft[6] = heading
        self.aircraft[7] = 250

        return self.postprocess_observation(self.aircraft)

    def preprocess_action(self, raw_action):
        action = np.array(raw_action).copy().reshape((3, self.num_aircraft))
        # -1 to 1 -> 1000 to 10000
        action[0] = (action[0] * 4500) + 5500
        # -1 to 1 -> -pi to pi
        action[1] *= np.pi
        # -1 to 1 -> 100 to 300
        action[2] = (action[2] * 100) + 200

        return action

    def postprocess_observation(self, raw_observation):

        observation = raw_observation.copy()[:5, :]
        # -30 to 30 -> -1 to 1
        observation[0] = observation[0] / 30
        # -30 to 30 -> -1 to 1
        observation[1] = observation[1] / 30
        # 1000 to 10_000 -> -1 to 1
        observation[2] = (observation[2] - 5500) / 4500
        # -np.pi to np.pi -> -1 to 1
        observation[3] = observation[3] / np.pi
        # 100 to 300 -> -1 to 1
        observation[4] = (observation[4] / 100) - 2

        # finally flatten
        return observation.flatten()

    def step(self, action):

        active_aircraft = self.aircraft[:, ~self.dones]

        active_aircraft[2] += (
            np.tanh(active_aircraft[5] - active_aircraft[2])
            * self.seconds_per_step
            * self.alt_rate_fpm
            / 60
        )
        active_aircraft[3] += normalize_angle(
            np.tanh(active_aircraft[6] - active_aircraft[3])
            * self.turn_rate_radps
            * self.seconds_per_step
        )
        active_aircraft[4] += (
            np.tanh(active_aircraft[7] - active_aircraft[4])
            * self.seconds_per_step
            * self.speed_rate_kps
        )

        moved = self.seconds_per_step * active_aircraft[4] / 3600
        active_aircraft[0] += moved * np.cos(active_aircraft[3])
        active_aircraft[1] += moved * np.sin(active_aircraft[3])

        actual_action = self.preprocess_action(action)
        active_aircraft[5:] = actual_action[:, ~self.dones]

        # score
        reward = self.passive_time_reward

        # check for ac leaving airspace
        reward += self.left_airspace_reward * np.sum(
            np.linalg.norm(active_aircraft[:2], axis=0) > (self.radius_nm + 2)
        )

        # check for collisions
        i, j = np.indices((len(active_aircraft[0]), len(active_aircraft[0])))
        distances = np.sqrt(
            (active_aircraft[0, i] - active_aircraft[0, j]) ** 2
            + (active_aircraft[0, i] - active_aircraft[0, j]) ** 2
        )
        amount_colliding = (
            np.sum(distances < self.minimum_separation_distance_nm) - self.num_aircraft
        )

        if amount_colliding > 0:
            reward += self.minimum_separation_distance_reward * amount_colliding

        # score altitude
        # at - 30x you should be at 7000, at +20 you should be at 2000, linear
        expected_altitude = -100 * (active_aircraft[0] - 40)
        reward += np.sum(
            np.abs(expected_altitude - active_aircraft[2])
            / 100
            * self.altitude_reward_per_100feet_off
            + 0.1
        )

        # score heading
        pos = active_aircraft[:2]
        a, b = self.favoured_heading(pos)
        he = np.arctan2(b, a)
        # this goes from -1 to 0
        reward += (
            -1
            * self.heading_off_reward
            * np.sum(-(np.abs(normalize_angle((he - active_aircraft[3]))) / np.pi) + 1)
        )

        # win conditions
        # if were between fap and fap + 3 from the origin
        # and if were between fap and fap + 3
        # and if heading is betweeen -pi+pi/4 and -pi/pi/4
        final_app_point = np.array([self.final_approach_point_nm, 0])

        new_dones = self.dones.copy()
        for i, done in enumerate(self.dones):
            if not done:
                correct1 = (
                    self.final_approach_point_nm
                    < np.linalg.norm(self.aircraft[:2, i], axis=0)
                ) & (
                    np.linalg.norm(self.aircraft[:2, i], axis=0)
                    < self.final_approach_point_nm + 3
                )

                correct2 = (
                    np.linalg.norm(self.aircraft[:2, i] - final_app_point, axis=0) < 3
                )
                correct3 = (3 / 4 * np.pi < self.aircraft[3, i]) | (
                    self.aircraft[3, i] < -3 / 4 * np.pi
                )
                if correct1 and correct2 and correct3:
                    reward += self.established_reward

                    new_dones[i] = True

        self.aircraft[:, ~self.dones] = active_aircraft
        self.dones = new_dones
        return self.postprocess_observation(self.aircraft), reward, self.dones.all(), {}

    def favoured_heading(self, vec):
        scaleX = 50
        scaleY = 100
        offsetX = self.final_approach_point_nm
        X = vec[0] - offsetX
        Y = vec[1]
        Ex = scaleX * (
            (X + 1) / ((X + 1) ** 2 + Y**2) - (X - 1) / ((X - 1) ** 2 + Y**2)
        )
        Ey = scaleY * (Y / ((X + 1) ** 2 + Y**2) - Y / ((X - 1) ** 2 + Y**2))

        # make uniform field for x < offset x
        if isinstance(Ex, np.ndarray):
            Ex[X < 0] = 1
            Ey[X < 0] = 0
        else:
            Ex = 1 if X < 1 else Ex
            Ey = 0 if X < 0 else Ey
        norm = np.sqrt(Ex**2 + Ey**2)
        return Ex / norm, Ey / norm

    def _xy_nm_to_pygame_coordinates(self, x, y) -> tuple:
        return (x / self.miles_per_pixel) + self.window_size / 2, -(
            y / self.miles_per_pixel
        ) + self.window_size / 2

    def render(self, **kwargs):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.font = pygame.font.SysFont(None, 24)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":

            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((255, 255, 255))

            # range rings
            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                (self.window_size / 2, self.window_size / 2),
                self.radius_nm / self.miles_per_pixel,
                2,
            )
            pygame.draw.circle(
                canvas,
                (100, 100, 100),
                (self.window_size / 2, self.window_size / 2),
                self.radius_nm / self.miles_per_pixel * 3 / 4,
                1,
            )
            pygame.draw.circle(
                canvas,
                (100, 100, 100),
                (self.window_size / 2, self.window_size / 2),
                self.radius_nm / self.miles_per_pixel * 2 / 4,
                1,
            )
            pygame.draw.circle(
                canvas,
                (100, 100, 100),
                (self.window_size / 2, self.window_size / 2),
                self.radius_nm / self.miles_per_pixel * 1 / 4,
                1,
            )

            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                self._xy_nm_to_pygame_coordinates(self.final_approach_point_nm, 0),
                3,
            )

            # localiser
            starts = self._xy_nm_to_pygame_coordinates(0, 0)
            ends = self._xy_nm_to_pygame_coordinates(
                self.localiser_range_nm * np.cos(0), self.localiser_range_nm * np.sin(0)
            )
            pygame.draw.line(canvas, (100, 0, 0), starts, ends)

            for ac in self.aircraft.T:
                coords = self._xy_nm_to_pygame_coordinates(ac[0], ac[1])
                # draw dot
                pygame.draw.circle(canvas, (0, 0, 255), coords, 4)

                # draw header lines
                header_end_x = (
                    ac[0] + np.cos(ac[3]) * ac[4] * self.header_line_length_nm * 3 / 60
                )
                header_end_y = (
                    ac[1] + np.sin(ac[3]) * ac[4] * self.header_line_length_nm * 3 / 60
                )

                pygame.draw.line(
                    canvas,
                    (0, 0, 255),
                    coords,
                    self._xy_nm_to_pygame_coordinates(header_end_x, header_end_y),
                    2,
                )

                x, y = np.meshgrid(np.linspace(-30, 30, 50), np.linspace(-30, 30, 50))

                favx, favy = self.favoured_heading(ac[:2])
                prefered_end_x = ac[0] + favx * 5
                prefered_end_y = ac[1] + favy * 5

                pygame.draw.line(
                    canvas,
                    (0, 255, 0),
                    coords,
                    self._xy_nm_to_pygame_coordinates(prefered_end_x, prefered_end_y),
                    2,
                )

                # draw tag
                if self.font:
                    canvas.blit(
                        self.font.render(f"{ac[2]}    {ac[4]}", False, (0, 0, 255)),
                        coords,
                    )

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
