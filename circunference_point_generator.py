import numpy as np

degrees = np.array([0, 20, 40, 60, 80, 90, 100, 110, 120, 140, 160, 180, 200, 220, 240, 260, 270, 290, 310, 330, 350])

radius_array = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.63, 0.65, 0.66]

multiple_circles = []

for radius in radius_array:
    circle = []
    for degree in degrees:
        circle.append(radius * np.cos(degree))
        circle.append(radius * np.sin(degree))
    multiple_circles.append(np.array(circle))


def get_points():
    return np.array(multiple_circles)
