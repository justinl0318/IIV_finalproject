import pygame

# Screen dimensions
WIDTH, HEIGHT = 1600, 800
CAR_WIDTH, CAR_HEIGHT = 200, 120
PEDESTRIAN_WIDTH, PEDESTRIAN_HEIGHT = 100, 140

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
        
        self.speed = 0 # Initial speed of the car
        self.max_speed = 3
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

class Pedestrian:
    def __init__(self, pedestrian_image):
        self.image = pygame.transform.scale(pedestrian_image, (PEDESTRIAN_WIDTH, PEDESTRIAN_HEIGHT))
        self.width, self.height = self.image.get_size()
        
        # Set initial position at the top center of the screen
        self.rect = self.image.get_rect()
        self.rect.topleft = ((WIDTH - self.width) // 2, 0)

        self.speed = 2

    def update(self):
        self.rect.y += self.speed

    def draw(self, screen):
        screen.blit(self.image, self.rect.topleft)