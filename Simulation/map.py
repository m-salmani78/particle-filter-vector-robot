import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import math
import shapely
from shapely.geometry import LineString, Point, Polygon


import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon

class Map:
    def __init__(self, map_file):
        self.objects = self.parse_map(map_file)
        self.add_offset()
        self.xmin, self.ymin, self.xmax, self.ymax = self.get_map_coordinates()
        print("$$$ MAP", self.xmin, self.ymin, self.xmax, self.ymax)

    def parse_map(self, map_file):
        tree = ET.parse(map_file)
        root = tree.getroot()
        objects = {}
        for model in root.findall("./world/model"):
            for child in model:
                if child.tag == "pose":
                    offset = child.text.split()
                    self.x_offset, self.y_offset = float(offset[0]), float(offset[1])

                if child.tag == "link":
                    pose = child.find("pose")  # Center
                    size = child.find("collision/geometry/box/size")  # Size
                    if pose is not None and size is not None:
                        name = child.attrib["name"]
                        xc, yc, z, roll, pitch, yaw = map(float, pose.text.split())  # Center of each cube (Pose)
                        w, h, _ = map(float, size.text.split())  # Width & Height (& z - length) of each cube
                        objects[name] = self.get_points(xc, yc, yaw, w, h)
        return objects

    def get_points(self, xc, yc, theta, w, h):
        upper_right = [
            xc + w * np.sin(theta) / 2 + h * np.cos(theta) / 2,
            yc + h * np.sin(theta) / 2 - w * np.cos(theta) / 2,
        ]
        upper_left = [
            xc - w * np.sin(theta) / 2 + h * np.cos(theta) / 2,
            yc + h * np.sin(theta) / 2 + w * np.cos(theta) / 2,
        ]
        lower_right = [
            xc + w * np.sin(theta) / 2 - h * np.cos(theta) / 2,
            yc - h * np.sin(theta) / 2 - w * np.cos(theta) / 2,
        ]
        lower_left = [
            xc - w * np.sin(theta) / 2 - h * np.cos(theta) / 2,
            yc - h * np.sin(theta) / 2 + w * np.cos(theta) / 2,
        ]
        return {
            "upper_right": upper_right,
            "upper_left": upper_left,
            "lower_right": lower_right,
            "lower_left": lower_left,
        }

    def add_offset(self):
        for obj in self.objects.values():
            for point in obj.values():
                point[0] += self.x_offset
                point[1] += self.y_offset

    def get_lines(self):
        lines = []
        for obj in self.objects.values():
            lines.append([obj["upper_right"], obj["lower_right"]])
            lines.append([obj["lower_right"], obj["lower_left"]])
            lines.append([obj["lower_left"], obj["upper_left"]])
            lines.append([obj["upper_left"], obj["upper_right"]])
        return lines

    def draw_map(self):
        for obj in self.objects.values():
            x_coords = [obj["upper_right"][0], obj["lower_right"][0], obj["lower_left"][0], obj["upper_left"][0]]
            y_coords = [obj["upper_right"][1], obj["lower_right"][1], obj["lower_left"][1], obj["upper_left"][1]]

            # Draw the edges of the object
            plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], c="black")

            # Fill the object
            plt.fill(x_coords, y_coords, facecolor="gray", edgecolor="black", linewidth=1)

    def point_in_object(self, x, y):
        point = Point(x, y)
        for obj in self.objects.values():
            points = [obj["upper_right"], obj["lower_right"], obj["lower_left"], obj["upper_left"]]
            poly = Polygon(points)
            if point.within(poly) or poly.touches(point):
                return True
        return False

    def get_map_coordinates(self):
        x = []
        y = []
        for obj in self.objects.values():
            for point in obj.values():
                x.append(point[0])
                y.append(point[1])
        return min(x), min(y), max(x), max(y)

    def point_in_map(self, x, y):
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def valid_point(self, x, y):
        return not self.point_in_object(x, y) and self.point_in_map(x, y)

    def find_intersection(self, point_1, point_2, point_3, point_4):
        line1 = LineString([tuple(point_1), tuple(point_2)])
        line2 = LineString([tuple(point_3), tuple(point_4)])
        intersection = line1.intersection(line2)
        return intersection

    def find_closest_intersection(self, x1, y1, x2, y2, max_distance):
        min_distance = max_distance
        for line in self.get_lines():
            intersection = self.find_intersection((x1, y1), (x2, y2), line[0], line[1])
            if intersection:
                distance = np.sqrt((intersection.x - x1) ** 2 + (intersection.y - y1) ** 2)
                if distance < min_distance:
                    min_distance = distance
        return min_distance

# Example usage:
# map_file = 'path_to_your_map_file.xml'
# my_map = Map(map_file)
