import pygame
import argparse
import cv2
import sys
import time
import math
import threading
from flask import Flask, request, jsonify
from object import Car, Pedestrian
from object import CAR_WIDTH, CAR_HEIGHT
from object import PEDESTRIAN_WIDTH, PEDESTRIAN_HEIGHT
from YOLO import model
from trajectory_prediction import weighted_moving_average

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1400, 800
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
GREEN = (0, 255, 0)  

# the number of pedestrian's future steps that the car should predict
PREDICT_STEPS = 100

# Frame rate
clock = pygame.time.Clock()
FPS = 60

# Initialize Flask app
app = Flask(__name__)

# Global dictionary to store precomputed_paths of each pedestrian
# key: id, value: {}
precomputed_paths = {}

# actively receive path from pedestrian
@app.route("/predict_trajectory", methods=["POST"])
def receive_future_path():
    global precomputed_paths

    data = request.json
    if not data or "pedestrian_id" not in data or "precomputed_path" not in data or "speed" not in data:
        return jsonify({"status": "failure", "message": "Invalid or empty data"}), 400
    
    pedestrian_id = data["pedestrian_id"]
    precomputed_path = data["precomputed_path"]
    speed = data["speed"]

    # store the information in the global dictionary
    precomputed_paths[pedestrian_id] = {
        "precomputed_path": precomputed_path,
        "speed": speed
    }

    return jsonify({"status": "success"}), 200
    

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

# active prediction of pedestrian trajectory
def car_control_logic_active(car: Car, pedestrians: list[Pedestrian], distance_threshold=200):

    # get the coordinate of car's head
    car_head = car.rect.midright
    car.decelerate_flag = False

    for pedestrian in pedestrians:
        if pedestrian.pedestrian_id not in precomputed_paths:
            continue

        if len(precomputed_paths[pedestrian.pedestrian_id]["precomputed_path"]) >= 2:
            precomputed_centered_path = [(x + pedestrian.width // 2, y) for x, y in precomputed_paths[pedestrian.pedestrian_id]["precomputed_path"]]
            pygame.draw.lines(screen, RED, False, precomputed_centered_path, 3)

# passive prediction of pedestrian trajectory
def car_control_logic_passive(car: Car, pedestrians: list[Pedestrian], xyxys, confidences, class_ids, distance_threshold=200):
    if len(xyxys) == 0:
        car.decelerate_flag = False
        return
    
    # get the coordinate of car's head
    car_head = car.rect.midright
    car.decelerate_flag = False

    for pedestrian in pedestrians:
        if len(pedestrian.trajectory) <= 10: break

        past_trajectory = pedestrian.trajectory[:] # copy
        future_trajectory = []
        for _ in range(PREDICT_STEPS):
            
            predicted_direction = weighted_moving_average(past_trajectory)
            predicted_step_x = past_trajectory[-1][0] + predicted_direction[0] * pedestrian.speed
            predicted_step_y = past_trajectory[-1][1] + predicted_direction[1] * pedestrian.speed
            
            past_trajectory.append((predicted_step_x, predicted_step_y))
            future_trajectory.append((predicted_step_x, predicted_step_y))
        
        # centered trajectory
        future_centered_trajectory = [(x + pedestrian.width // 2, y) for x, y in future_trajectory]
        pygame.draw.lines(screen, RED, False, future_centered_trajectory, 2)

    # for index, xyxy in enumerate(xyxys):
    #     if class_ids[index] == 2: # skip car
    #         continue

    #     for pedestrian in pedestrians:
    #         topleft = (int(xyxy[0]), int(xyxy[1]))
    #         # get mid left coordinate of pedestrian
    #         midleft = (topleft[0], topleft[1] + (PEDESTRIAN_HEIGHT // 2)) 

    #         distance = get_distance(car_head, midleft)
    #         print(f"Distance: {distance}, Threshold: {distance_threshold}")  # Debug print
    #         if distance <= distance_threshold and pedestrian.entering is True:
    #             car.decelerate_flag = True
    #             break


def main(flag: bool, granularity_size: int, n_rounds: int):
    running = True # game loop
    car = Car(CAR_IMAGE)
    num_pedestrian = 1
    pedestrians = [Pedestrian(PEDESTRIAN_IMAGE, id=i) for i in range(num_pedestrian)]
    
    if flag == "active":
        # start Flask server in a separate thread
        threading.Thread(target=app.run, kwargs={"debug": False, "host": "0.0.0.0", "port": 5000}).start()

    rounds = 0
    paused = 0 # even = false, odd = true

    while running:
        if rounds >= n_rounds:
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = (paused + 1) % 2

        # Capture pygame screen
        screenshot = pygame.surfarray.array3d(screen)
        screenshot = cv2.transpose(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # feed image into YOLO model
        xyxys, confidences, class_ids = predict(screenshot)
 
        # Check if the car reaches the end of the frame
        # if yes, start a new round
        if car.rect.x + car.width >= WIDTH:
            rounds += 1
            car.start_new_round()
            for pedestrian in pedestrians:
                pedestrian.start_new_round()


        if paused % 2 == 0:
            # clear screen first
            screen.fill(WHITE)

            if flag == "active":
                car_control_logic_active(car, pedestrians)
            elif flag == "passive":
                car_control_logic_passive(car, pedestrians, xyxys, confidences, class_ids)    
            
            # move the car
            car.update()
            print(car.decelerate_flag)
            
            # Move pedestrian
            for pedestrian in pedestrians:
                pedestrian.update()

                if flag == "active":
                    pedestrian.send_trajectory_to_car()

                # Check if the pedestrian is leaving the intersection
                if pedestrian.rect.y >= HEIGHT // 2:
                    pedestrian.entering = False
  
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
            for pedestrian in pedestrians:
                pedestrian.draw(screen) # draw pedestrian
            pygame.display.flip()
            clock.tick(FPS)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Active pedestrian detection")
    parser.add_argument("--flag", type=str, default="passive", choices=["active", "passive"], help="active or passive pedestrian detection")
    parser.add_argument("--granularity_size", type=int, default=10, help="Granularity size for collision detection")
    parser.add_argument("--n_rounds", type=int, default=5, help="Number of rounds to run")
    args = parser.parse_args()

    flag = args.flag
    granularity_size = args.granularity_size
    n_rounds = args.n_rounds
    main(flag, granularity_size, n_rounds)