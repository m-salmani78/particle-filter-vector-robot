import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import math
from shapely.geometry import LineString, Point, Polygon

class Map:
    def __init__(self, address):
        self.rects, self.global_map_pose, self.map_boundry = self.init_map(address)
        self.x0 , self.y0 = float(self.global_map_pose[0]) , float(self.global_map_pose[1])
        self.rects = self.add_offset(self.rects,[self.x0 , self.y0])
        self.all_map_lines = self.convert_point_to_line()
        self.polygans = self.convert_to_polygons()

    def init_map(self, address):
        tree = ET.parse(address)
        root = tree.getroot()

        rects = []
        centers = []

        for links in root[0].iter("model"):
            try:
                for link in links:
                    if link.tag == "pose":
                        _global_map_pose = link.text.split(" ")
                        if _global_map_pose[0] != "0":
                            global_map_pose = _global_map_pose
                            print(f'global_map_pose:\n{global_map_pose}')

                    if link.tag == "link":
                        geometry = None
                        pose = None
                        for _pose in link.iter("pose"):
                            pose = _pose.text.split(" ")
                            break
                        for collision in link.iter("collision"):
                            geometry = collision[3][0][0].text.split(" ")

                        p1 = [
                            float(pose[0])
                            + (float(geometry[0]) * math.cos(float(pose[5])) / 2)
                            + float(geometry[1]) * math.sin(float(pose[5])) / 2,
                            float(pose[1])
                            + float(geometry[0]) * math.sin(float(pose[5])) / 2
                            - float(geometry[1]) * math.cos(float(pose[5])) / 2,
                        ]

                        p2 = [
                            float(pose[0])
                            + float(geometry[0]) * math.cos(float(pose[5])) / 2
                            - float(geometry[1]) * math.sin(float(pose[5])) / 2,
                            float(pose[1])
                            + float(geometry[0]) * math.sin(float(pose[5])) / 2
                            + float(geometry[1]) * math.cos(float(pose[5])) / 2,
                        ]

                        p3 = [
                            float(pose[0])
                            - float(geometry[0]) * math.cos(float(pose[5])) / 2
                            + float(geometry[1]) * math.sin(float(pose[5])) / 2,
                            float(pose[1])
                            - float(geometry[0]) * math.sin(float(pose[5])) / 2
                            - float(geometry[1]) * math.cos(float(pose[5])) / 2,
                        ]

                        p4 = [
                            float(pose[0])
                            - float(geometry[0]) * math.cos(float(pose[5])) / 2
                            - float(geometry[1]) * math.sin(float(pose[5])) / 2,
                            float(pose[1])
                            - float(geometry[0]) * math.sin(float(pose[5])) / 2
                            + float(geometry[1]) * math.cos(float(pose[5])) / 2,
                        ]

                        rects.append([p1, p2, p3, p4])
                        centers.append([pose[0], pose[1]])
            except Exception as e:
                pass

        return rects, global_map_pose, self.get_map_boundry(centers)

    def find_intersection(self, p1, p2, p3, p4):
        line1 = LineString([tuple(p1), tuple(p2)])
        line2 = LineString([tuple(p3), tuple(p4)])

        int_pt = line1.intersection(line2)
        if int_pt:
            point_of_intersection = int_pt.x, int_pt.y
            return point_of_intersection
        else:
            return False

    def convert_point_to_line(self):
        lines = []
        for points in self.rects:
            lines.append([points[0], points[1]])
            lines.append([points[1], points[3]])
            lines.append([points[3], points[2]])
            lines.append([points[2], points[0]])
        return lines

    def add_offset(self, rects, offset):
        new_rects = []
        for points in rects:
            new_rects.append(
                [
                    [points[0][0] + offset[0], points[0][1] + offset[1]],
                    [points[1][0] + offset[0], points[1][1] + offset[1]],
                    [points[2][0] + offset[0], points[2][1] + offset[1]],
                    [points[3][0] + offset[0], points[3][1] + offset[1]],
                ]
            )
        return new_rects

    def convert_to_polygons(self):
        polygons: list[Polygon] = []
        for points in self.rects:
            polygons.append(
                Polygon(
                    [tuple(points[0]), tuple(points[1]), tuple(points[3]), tuple(points[2])]
                )
            )
        return polygons

    def check_is_collition(self, point):
        p = Point(tuple(point))
        for rect in self.polygans:
            if rect.contains(p):
                return True
        return False

    def get_map_boundry(self, centers):
        X = []
        Y = []

        for item in centers:
            X.append(float(item[0]))
            Y.append(float(item[1]))

        return min(X), max(X), min(Y), max(Y)

    def out_of_range(self, particle, offset):
        if (
            particle[0] - offset[0] > self.map_boundry[1]
            or particle[0] - offset[0] < self.map_boundry[0]
        ):
            return True
        elif (
            particle[1] - offset[1] > self.map_boundry[3]
            or particle[1] - offset[1] < self.map_boundry[2]
        ):
            return True
        else:
            return False

    def plot_map(self):
        for rect in self.all_map_lines:
            rect = list(zip(*rect))
            plt.plot(rect[1], rect[0], c="black")
