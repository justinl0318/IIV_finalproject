import pygame
import cv2
import sys
import time
import math
from object import Car, Pedestrian
from object import CAR_WIDTH, CAR_HEIGHT
from object import PEDESTRIAN_WIDTH, PEDESTRIAN_HEIGHT
from YOLO import model

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car and Pedestrian Simulation")

# Load images
CAR_IMAGE = pygame.image.load("car.jpg").convert_alpha()
PEDESTRIAN_IMAGE = pygame.image.load("pedestrian.jpg").convert_alpha()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)  # Color for bounding boxes

# Frame rate
clock = pygame.time.Clock()
FPS = 60

def get_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def predict(screenshot):
    results = model(screenshot)
    xyxys = []
    confidences = []
    class_ids = []

    for result in results: 
        boxes = result.boxes.cpu().numpy()
        for xyxy in boxes.xyxy: # iterate over each bounding box in the current frame
            xyxys.append(xyxy)
        for conf in boxes.conf:
            confidences.append(conf)
        for cls in boxes.cls:
            class_ids.append(cls)

    return xyxys, confidences, class_ids

def car_control_logic(car: Car, pedestrian: Pedestrian, xyxys, confidences, class_ids, distance_threshold=300):
    if len(xyxys) == 0:
        car.decelerate_flag = False
        return
    
    # get the coordinate of car's head
    car_head = car.rect.midright
    
    for index, xyxy in enumerate(xyxys):
        if class_ids[index] == 2: # skip car
            continue

        topleft = (int(xyxy[0]), int(xyxy[1]))
        # get mid left coordinate of pedestrian
        midleft = (topleft[0], topleft[1] + (PEDESTRIAN_HEIGHT // 2)) 

        distance = get_distance(car_head, midleft)
        print(f"Distance: {distance}, Threshold: {distance_threshold}")  # Debug print
        if distance <= distance_threshold and pedestrian.entering is True:
            car.decelerate_flag = True
        else:
            car.decelerate_flag = False

def main():
    running = True # game loop
    car = Car(CAR_IMAGE)
    pedestrian = Pedestrian(PEDESTRIAN_IMAGE)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Capture pygame screen
        screenshot = pygame.surfarray.array3d(screen)
        screenshot = cv2.transpose(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # feed image into YOLO model
        xyxys, confidences, class_ids = predict(screenshot)

        # Check if the car reaches the end of the frame
        # if yes, start a new round
        paused = False
        if car.rect.x + car.width >= WIDTH:
            paused = True
            while paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        paused = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            # Start a new round
                            car.rect.x = 0
                            pedestrian.rect.y = 0
                            pedestrian.entering = True
                            paused = False

        if not paused:
            car_control_logic(car, pedestrian, xyxys, confidences, class_ids)
            # move the car
            car.update()
            print(car.decelerate_flag)
            # Move pedestrian
            pedestrian.update()
            # Check if the pedestrian is leaving the intersection
            if pedestrian.rect.y >= HEIGHT // 2:
                pedestrian.entering = False

            screen.fill(WHITE)

            # draw bounding box
            if xyxys:
                for index, xyxy in enumerate(xyxys):
                    # dont need to draw boundary for car
                    if class_ids[index] == 2: # id = 2: car
                        continue
                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    pygame.draw.rect(screen, GREEN, pygame.Rect((x1-28, y1-10), (100+7, 140+25)), width=2)

            car.draw(screen) # draw car
            pedestrian.draw(screen) # draw pedestrian
            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    main()




