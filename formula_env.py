import numpy as np
import gym
from  tracks.circular import circular_path
from shapely.geometry import Polygon, Point
import pygame

INF  = 1e4

class FormulaEnv(gym.Env):
    def __init__(self, render_mode=None, render_sleep=100):

        self.dt = 0.1 # time step
        self.wheel_base = 1.5 # wheel base
        self.steer_limit = np.deg2rad(75) # steering angle limit
        self.cones_num = 6 # number of cones
        
        [self.left_cones,
            self.right_cones,
            self.start_cones,
            self.start_point] = circular_path(radius=5.0, num_points=100, track_width=2.0)
        
        self.agent_state = np.array([self.start_point[0], self.start_point[1], self.start_point[2], 0.0, 0.0]) 
        # states -> x , y, yaw, velocity, steer_angle
        
        self.observation_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(low=-INF, high=INF, shape=(2,), dtype=np.float32),
                "velocity": gym.spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float32),
                "yaw": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                "steer_angle": gym.spaces.Box(low=-self.steer_limit, high=self.steer_limit, shape=(1,), dtype=np.float32),
                "left_cones": gym.spaces.Box(low=-INF, high=INF, shape=(self.cones_num, 2), dtype=np.float32),
                "right_cones": gym.spaces.Box(low=-INF, high=INF, shape=(self.cones_num, 2), dtype=np.float32),
                "start_cones": gym.spaces.Box(low=-INF, high=INF, shape=(2, 2), dtype=np.float32),
            }
        )
        
        self.action_space = gym.spaces.Dict(
            {
                "steering_velocity": gym.spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(1,), dtype=np.float32),
                "acceleration": gym.spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32),
            }
        )
        
        self.state_dim = sum(np.prod(space.shape) for space in self.observation_space.spaces.values())
        self.action_dim = sum(np.prod(space.shape) for space in self.action_space.spaces.values())
        
        self.render_mode = render_mode
        if render_mode is not None:
            self.render_sleep = render_sleep
            pygame.init()
            self.screen_width = 1000
            self.screen_height = 1000
            self.stats_width = 200
            self.stats_height = self.screen_height
            self.screen = pygame.display.set_mode((self.screen_width + self.stats_width, self.screen_height))
            self.sim_screen = pygame.Surface((self.screen_width, self.screen_height))
            self.stats_screen = pygame.Surface((self.stats_width, self.stats_height))
            self.stats_screen.fill((0, 0, 0))
            self.screen.blit(self.sim_screen, (0, 0))
            self.screen.blit(self.stats_screen, (self.screen_width, 0))
            pygame.display.set_caption("Formula Car Environment")
            self.clock = pygame.time.Clock()
            self._render(reward=0)
    
    def _car_kinematic(self, state, action):
        """
        Update the car's state based on the action taken.
        """
        new_state = state.copy()
        
        x, y, yaw, velocity, steer_angle = state
        steer_velocity = action[1]
        acceleration = action[0]
        
        # Update the car's state using simple kinematic equations
        new_state[0] += velocity * np.cos(yaw) * self.dt
        new_state[1] += velocity * np.sin(yaw) * self.dt
        new_state[2] += (velocity / 2.5) * np.tan(steer_angle) * self.dt
        new_state[3] += acceleration * self.dt
        new_state[4] = np.clip(new_state[4] + steer_velocity*self.dt , -self.steer_limit, self.steer_limit)  
        
        return new_state
    
    
    def _return_observation(self):
        """
        Return the current observation of the environment.
        """
        
        left_cones_dist = np.linalg.norm(self.left_cones - self.agent_state[:2], axis=1)
        argmin_left_cones = np.argsort(left_cones_dist)
        close_left_cones = self.left_cones[argmin_left_cones][:self.cones_num]
        
        right_cones_dist = np.linalg.norm(self.right_cones - self.agent_state[:2], axis=1)
        argmin_right_cones = np.argsort(right_cones_dist)
        close_right_cones = self.right_cones[argmin_right_cones][:self.cones_num]
        
        orange_cones_dist = np.linalg.norm(self.start_cones - self.agent_state[:2], axis=1)
        if orange_cones_dist[0] < 5 and orange_cones_dist[1] < 5:
            close_start_cones = self.start_cones
        else:
            close_start_cones = np.array([[-INF, -INF], [-INF, -INF]]) 
        
       # return as a line array
        return np.array([
            self.agent_state[0],
            self.agent_state[1],
            self.agent_state[2],
            self.agent_state[3],
            self.agent_state[4]] +
            close_left_cones.flatten().tolist() +
            close_right_cones.flatten().tolist() +
            close_start_cones.flatten().tolist()
        )
        
        
    def _get_reward(self, observation):
        """
        Calculate the reward based on the current state of the agent.
        """        
        reward = 0.0
        
        left_cones = observation[5:5+self.cones_num*2].reshape(self.cones_num, 2)
        right_cones = observation[5+self.cones_num*2:5+self.cones_num*4].reshape(self.cones_num, 2)
        
        # check if the car is within the track boundaries
        polygon = Polygon( left_cones.tolist() + right_cones.tolist())
        position = observation[:2]
        if not polygon.contains(Point(position)):
            print("Out of track")
            reward += -10.0

        velocity = observation[3]
        if velocity < 0.1:
            print("Car is not moving")
            reward += -1.0
            
        reward += velocity * 0.1
        print(velocity)
        return reward
        
    def reset(self):
        """
        Reset the environment to an initial state.
        """
        self.agent_state = np.array([self.start_point[0], self.start_point[1], self.start_point[2], 0.0, 0.0])
        return self._return_observation()
        
    def step(self, action):
        """
        Take a step in the environment based on the action taken.
        """
        self.agent_state = self._car_kinematic(self.agent_state, action)
        observation = self._return_observation()
        reward = self._get_reward(observation)
        
        done = False
        
        if self.render_mode is not None:
            self._render(reward)

        return observation, reward, done, {}, {}

    def _render(self, reward):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        
        self.screen.fill((255, 255, 255))
        

        # Draw left cones
        for cone in self.left_cones:
            pygame.draw.circle(self.screen, (0, 0, 255), (int(cone[0] * 80 + self.screen_width / 2), int(-cone[1] * 80 + self.screen_height / 2)), 5)

        # Draw right cones
        for cone in self.right_cones:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(cone[0] * 80 + self.screen_width / 2), int(-cone[1] * 80 + self.screen_height / 2)), 5)

        # Draw start cones
        for cone in self.start_cones.T:
            pygame.draw.circle(self.screen, (255, 165, 0), (int(cone[0] * 80 + self.screen_width / 2), int(-cone[1] * 80 + self.screen_height / 2)), 5)
            
        # Draw car's position
        car_surface = pygame.Surface((40, 20), pygame.SRCALPHA)  # Create a transparent surface
        car_surface.fill((255, 0, 0))  # Fill the car surface with red (or any color)

        # Rotate the car surface
        if(np.isnan(self.agent_state[2])):
            self.agent_state[2] = 0
        rotated_car = pygame.transform.rotate(car_surface, self.agent_state[2] * 180 / np.pi)

        # Get the rotated car's rect and set its center
        rotated_car_rect = rotated_car.get_rect(center=(int(self.agent_state[0] * 80 + self.screen_width / 2), 
                                                        int(-self.agent_state[1] * 80 + self.screen_height / 2)))

        # Blit the rotated car onto the screen
        self.screen.blit(rotated_car, rotated_car_rect)

        # Update the stats screen (right part of the screen)
        self.stats_screen.fill((50, 50, 50))  # Fill the stats screen with green

        # Display the reward as a number
        font = pygame.font.Font(None, 36)  # Create a font object
        reward_text = font.render(f"Reward: {reward:.2f}", True, (0, 0, 0))  # Render the reward text
        self.stats_screen.blit(reward_text, (10, 10))  # Blit the text onto the stats screen

        # Draw a bar representing the reward
        bar_width = int((reward / 100) * (self.stats_width - 20))  # Scale the reward to fit the bar width
        bar_width = max(0, min(bar_width, self.stats_width - 20))  # Clamp the bar width between 0 and max width
        pygame.draw.rect(self.stats_screen, (0, 0, 0), (10, 50, bar_width, 20))  # Draw the reward bar

        # Blit the stats screen onto the main screen
        self.screen.blit(self.stats_screen, (self.screen_width, 0))

        
        pygame.display.flip()
        self.clock.tick(60)
        pygame.time.delay(self.render_sleep)
        
    def close(self):
        """
        Close the environment and clean up resources.
        """
        if self.render_mode is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.render_mode = None
        self.agent_state = None
        self.left_cones = None
        self.right_cones = None
        self.start_cones = None
        self.start_point = None
        self.observation_space = None
        self.action_space = None
        self.dt = None
        self.wheel_base = None
        self.steer_limit = None
        
