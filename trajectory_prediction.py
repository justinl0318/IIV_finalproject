import math
import numpy as np

# the number of past trajectory coordinates that should be taken 
# into consideration to compute weighted moving average
ACCOUNTED_LENGTH = 50

def weighted_moving_average(trajectory: list):

    # calculate direction vectors and corresponding angles
    direction_vectors = []
    angles = []

    accounted_length = min(len(trajectory), ACCOUNTED_LENGTH)
    
    # determine start index
    start_index = max(0, len(trajectory) - accounted_length - 1)

    for i in range(start_index, len(trajectory) - 1):
        dx = trajectory[i+1][0] - trajectory[i][0]
        dy = trajectory[i+1][1] - trajectory[i][1]

        direction_vectors.append((dx, dy))
        angle = math.atan2(dy, dx)
        angles.append(angle)

    length = len(angles)
    weights = np.arange(1, length + 1)

    # calculate weighted moving average of the angles
    weighted_sum = sum(w * angle for w, angle in zip(weights, angles))
    sum_of_weights = sum(weights)
    theta_WMA = weighted_sum / sum_of_weights   

    # convert the average angle to a unit direction vector for the predicted direction
    predicted_direction = (math.cos(theta_WMA), math.sin(theta_WMA))
    return predicted_direction