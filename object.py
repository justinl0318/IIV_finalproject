import pygame
import random
import math
import requests

# Screen dimensions
WIDTH, HEIGHT = 1600, 800
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
        
        self.max_speed = 10
        self.speed = self.max_speed
        self.acceleration = 0.3
        self.deceleration = 0.6

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

        self.speed = 4

        # The pedestrian is entering the intersection initially
        # Update this flag after he pass the middle of the screen
        self.entering = True

        self.start_new_round()
        

    def start_new_round(self):
        self.entering = True

        # Set the initial position 
        initial_x = random.randint(0, WIDTH - self.width)
        self.rect.topleft = (initial_x, 0)
        self.initial_pos = (self.rect.x, self.rect.y)

        self.make_u_turn = random.random() > 0.5
        # Set the final target position 
        if self.make_u_turn == True:
            self.final_pos = (self.initial_pos[0] + random.randint(-100, 100), 0)
        else:     
            self.final_pos = (self.initial_pos[0] + random.randint(-100, 100), HEIGHT)

        # Generate intermediate waypoints
        self.waypoints = self.generate_waypoints(self.initial_pos, self.final_pos)
        self.current_waypoint_index = 0

        # precompute the path based on waypoints
        self.path = self.compute_path()
        self.path_index = 0

        # store past trajectory
        self.trajectory = []
        
    def generate_waypoints(self, start, end):
        waypoints = [start]
        num_turns = random.randint(2, 5)
        min_x = min(self.initial_pos[0], self.final_pos[0])
        max_x = max(self.initial_pos[0], self.final_pos[0])
        prev_y = start[1]
        if self.make_u_turn == True:
            # the y coordinate at which the pedestrian will make a u turn
            middle_y = (HEIGHT - self.height) // 2 - 70
            # go down
            for _ in range(num_turns // 2):
                waypoints_x = random.randint(min_x, max_x)
                waypoints_y = random.randint(prev_y, middle_y)
                waypoints.append((waypoints_x, waypoints_y))

                prev_y = waypoints_y # update prev_y

            waypoints.append((waypoints[-1][0], middle_y))

            # go up
            prev_y = middle_y
            for _ in range(num_turns // 2, num_turns):
                waypoints_x = random.randint(min_x, max_x)
                waypoints_y = random.randint(0, prev_y)
                waypoints.append((waypoints_x, waypoints_y))

                prev_y = waypoints_y
            
            waypoints.append(end)
        else:
            for _ in range(num_turns):
                waypoints_x = random.randint(min_x, max_x)
                waypoints_y = random.randint(prev_y, self.final_pos[1])
                waypoints.append((waypoints_x, waypoints_y))
                
                prev_y = waypoints_y # update prev_y
                
            waypoints.append(end)
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
        