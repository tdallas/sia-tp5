import numpy as np

degrees = np.linspace(0, 360, 21)

radius_array = np.linspace(0.1, 0.9, 100)

multiple_circles = []

for radius in radius_array:
    circle = []
    for degree in degrees:
        circle.append(radius * np.cos(degree))
        circle.append(radius * np.sin(degree))
    multiple_circles.append(np.array(circle))


def get_points():
    print(radius_array)
    return np.array(multiple_circles)
