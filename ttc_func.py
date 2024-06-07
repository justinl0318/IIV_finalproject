# ttc_func.py
import math
from object import Car, Pedestrian

def get_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_intersection_point(car: Car, path):
    for coord in path:
        # print("hello, coord: " + str(coord[1] // 1) + "rect.y: " + str(car.rect.y))

        if -9 < coord[1] - car.rect.y and coord[1] - car.rect.y < 9:
            return coord

    return None

def calculate_elapsed_time(d, v0, a, v_max):
    # Time to reach maximum speed
    t_acc = (v_max - v0) / a
    
    # Distance covered to reach maximum speed
    d_acc = v0 * t_acc + 0.5 * a * t_acc**2
    
    if d_acc >= d:
        # The car reaches the finish line while still accelerating
        discriminant = v0**2 + 2 * a * d
        if discriminant < 0:
            return None  # No real solution, car can't reach the finish line
        t = (-v0 + math.sqrt(discriminant)) / a
        return t
    else:
        # The car reaches maximum speed and then travels at constant speed
        d_const = d - d_acc
        t_const = d_const / v_max
        t_total = t_acc + t_const
        return t_total

def calculate_ttc(car: Car, pedestrian: Pedestrian, path):
    pos = find_intersection_point(car, path)

    # ignore the pedestrians who would not collide w/ the car or is behide the car
    if pos is None or pos[0] <= car.rect.x:
        return -1, -1, (-1, -1)
    
    # take acceleration & max speed into account
    car_ttc = calculate_elapsed_time((pos[0] - car.rect.x), car.speed, car.acceleration, car.max_speed) 

    

    pedestrian_ttc = 0
    for coord in path:
        pedestrian_ttc += 1
        if coord == pos:
            break

    # print("CARTTC: ", car_ttc, "PEDESTRIAN_TTC: ", pedestrian_ttc)
    return car_ttc, pedestrian_ttc, pos

