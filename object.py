import pygame
import random
import math
import requests

# Screen dimensions
WIDTH, HEIGHT = 1400, 800
CAR_WIDTH, CAR_HEIGHT = 200, 120
PEDESTRIAN_WIDTH, PEDESTRIAN_HEIGHT = 100, 140

# Server URL
SERVER_URL = 'http://127.0.0.1:5000/predict_trajectory'

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class Car:
    def __init__(self, car_image):
        # scale: (widt, height)
        self.image = pygame.transform.scale(car_image, (CAR_WIDTH, CAR_HEIGHT))
        self.width, self.height = self.image.get_size()

        self.rect = self.image.get_rect()
        self.rect.topleft = (0, (HEIGHT - self.height) // 2)
        
        self.max_speed = 12
        self.speed = self.max_speed
        self.acceleration = 1
        self.deceleration = 2

        self.decelerate_flag = False

    def update(self):
        if self.decelerate_flag == False:
            if self.speed < self.max_speed:
                self.speed += self.acceleration
            elif self.speed >= self.max_speed: # set limit
                self.speed = self.max_speed

        else:
            if self.speed > 0:
                self.speed -= self.deceleration
            elif self.speed <= 0: # set limit
                self.speed = 0
            
        self.rect.x += self.speed

    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)

    def start_new_round(self):
        self.rect.x = 0


class Pedestrian:
    def __init__(self, pedestrian_image, id=0):
        self.pedestrian_id = id
        self.image = pygame.transform.scale(pedestrian_image, (PEDESTRIAN_WIDTH, PEDESTRIAN_HEIGHT))
        self.width, self.height = self.image.get_size()
        
        self.rect = self.image.get_rect()
        self.start = ((WIDTH - self.width) // 2, 0)
        self.rect.topleft = self.start

        self.speed = 9

        self.case = random.randrange(4)

        # The pedestrian is entering the intersection initially
        # Update this flag after he pass the middle of the screen
        self.entering = True

        self.start_new_round()
        

    def start_new_round(self):
        self.entering = True

        # Generate intermediate waypoints
        self.waypoints = self.generate_waypoints()
        self.current_waypoint_index = 0

        # precompute the path based on waypoints
        self.path = self.compute_path()
        self.path_index = 0

        # store past trajectory
        self.trajectory = []
        
    def generate_waypoints(self):
        waypoints = [self.start]
        middle_x = (WIDTH - self.width) // 2
        middle_y = HEIGHT // 2
        end_y = HEIGHT

        if self.case == 0: # straight line
            self.end = (middle_x, end_y)
            waypoints.append(self.end)

        elif self.case == 1: # go right a little bit and go back to left
            self.mid = (middle_x + 200, middle_y - 100)
            self.end = (middle_x, end_y)
            waypoints.append(self.mid)
            waypoints.append(self.end)

        elif self.case == 2: # bus stop case
            self.mid = (middle_x, middle_y - 100)
            self.end = (middle_x - 200, end_y)
            waypoints.append(self.mid)
            waypoints.append(self.end)

        elif self.case == 3: # u turn
            self.mid = (middle_x, middle_y - 100)
            self.end = (middle_x, 0)
            waypoints.append(self.mid)
            waypoints.append(self.end)
        
        return waypoints
    
    def compute_path(self):
        # precompute path from start to end based on the waypoints
        path = []
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]
            end = self.waypoints[i+1]
            distance = math.hypot(end[0] - start[0], end[1] - start[1]) 
            steps = int(distance // self.speed)
            for step in range(steps):
                t = step / steps
                x = int(start[0] + t * (end[0] - start[0]))
                y = int(start[1] + t * (end[1] - start[1]))
                path.append((x, y))
        
        path.append(self.waypoints[-1]) # include the final waypoints
        return path
    
    def update(self):
        # update pedestrian movement
        if self.path_index < len(self.path):
            self.rect.topleft = self.path[self.path_index]
            self.path_index += 1
            self.trajectory.append((self.rect.x, self.rect.y))

    def draw(self, screen):
        # draw the trajectory line
        if len(self.path) > 1:
            centered_path = [(x + self.width // 2, y) for x, y in self.path]
            pygame.draw.lines(screen, BLUE, False, centered_path, 2)

        screen.blit(self.image, self.rect.topleft)

    def send_trajectory_to_car(self):
        # actively sends the trajectory to the car
        response = requests.post(SERVER_URL, json={
            "pedestrian_id": self.pedestrian_id,
            "precomputed_path": self.path[self.path_index:],
            "speed": self.speed
        })
        