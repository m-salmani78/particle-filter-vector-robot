#! /usr/bin/env python3
import rospy
from tf.transformations import euler_from_quaternion
import mapFromExcel as map
import numpy as np
from math import *
from os.path import expanduser
import matplotlib.pyplot as plt
import scipy
from anki_vector_ros.msg import Drive, Proximity
import time
import scipy
from sklearn.cluster import DBSCAN

PI = 3.1415926535897
## Constants and parameters :
num_prtcls = 1000


distance_sensor_model = {
    0: {"mean": 0.0, "std": 0.01},
    5: {"mean": -0.0011, "std": 0.033},
    10: {"mean": 0.0001, "std": 0.023},
    20: {"mean": -0.0013, "std": 0.074},
    30: {"mean": -0.0148, "std": 0.092},
    37: {"mean": -0.0213, "std": 0.138},
}
translation_std = 2 * (0.02834) ** 0.5 / 100
rotate_std = 2 * 0.3 * (1.7283) ** 0.5 * PI / 180
g_std = 2 * (0.01349) / 100


x = 0.0
y = 0.0
theta = 0.0


laser_data = 0
laser = 0

desired_time = 5  # dt is fixed but we calculate speed

rotation_list = [PI / 2, 3 * (PI / 2), 0, PI]

home = expanduser("~")
map_address = home + "/catkin_ws/src/Vector-Robot-Localization/Real World/mapworld.xlsx"
rects, global_map_pose, map_boundry = map.init_map(map_address)
x_min, x_max, y_min, y_max = map_boundry
x0, y0 = float(global_map_pose[0]), float(global_map_pose[1])
all_map_lines = map.convert_point_to_line(rects)

rects = map.add_offset(rects, [x0, y0])
all_map_lines = map.convert_point_to_line(rects)
polygan = map.convert_to_poly(rects)


# Functions :


def callback_laser(msg):
    global laser
    laser = (msg.distance) / 1000


def motion_model(
    prtcl_weight,
    v,
    w,
    dt,
    v_var=translation_std,
    w_var=rotate_std,
    g_var=g_std,
    map=map,
    polygan=polygan,
    x0=x0,
    y0=y0,
    map_boundry=map_boundry,
):

    for i in range(prtcl_weight.shape[0]):
        v_err = 0
        w_err = 0

        if v > 0:
            v_err = 0.012 / dt
        elif v < 0:
            v_err = -0.012 / dt

        if w > 0:
            w_err = 2 * (2.3 * PI / 180) / dt
        elif w < 0:
            w_err = -2 * (2.3 * PI / 180) / dt

        v_hat = v + np.random.normal(-v_err, v_var)
        w_hat = w + np.random.normal(w_err, w_var)
        if w_hat == 0:
            w_hat = 1e-9
        g_hat = np.random.normal(0, g_var)

        prtcl_weight[:, 0][i] = prtcl_weight[:, 0][i] + (v_hat / w_hat) * (
            -sin(prtcl_weight[:, 2][i]) + sin(prtcl_weight[:, 2][i] + w_hat * dt)
        )

        prtcl_weight[:, 1][i] = prtcl_weight[:, 1][i] + (v_hat / w_hat) * (
            cos(prtcl_weight[:, 2][i]) - cos(prtcl_weight[:, 2][i] + w_hat * dt)
        )

        prtcl_weight[:, 2][i] = prtcl_weight[:, 2][i] + dt * (w_hat + g_hat) + 10 * PI
        prtcl_weight[:, 2][i] = (
            prtcl_weight[:, 2][i] - int(prtcl_weight[:, 2][i] / (2 * PI)) * 2 * PI
        )

    return prtcl_weight


def measurment_model(
    partcle_params, z, map=map, all_map_lines=all_map_lines
):
    max_laser = 0.4

    for i in range(partcle_params.shape[0]):
        prtcl_start = [partcle_params[:, 0][i], partcle_params[:, 1][i]]
        prtcl_end = [
            partcle_params[:, 0][i] + max_laser * cos(partcle_params[:, 2][i]),
            partcle_params[:, 1][i] + max_laser * sin(partcle_params[:, 2][i]),
        ]
        min_distance = 0.05
        col = False

        for line in all_map_lines:
            intersection_point = map.find_intersection(
                line[0], line[1], prtcl_start, prtcl_end
            )
            if intersection_point != False:
                d_laser = (
                    (intersection_point[0] - partcle_params[:, 0][i]) ** 2
                    + (intersection_point[1] - partcle_params[:, 1][i]) ** 2
                ) ** 0.5
                col=True
                if min_distance >= d_laser:
                    d_laser = min_distance
                    # col = intersection_point
                

        if col == False:
            d_laser = max_laser

        z_type=5
        if z < 0.07:
            z_type = 5
        elif z > 0.34:
            z_type = 37
        else:
            z_type = round(z*10) * 10

        laser_nois = distance_sensor_model[z_type]
        print(f'z={z}, z type={z_type}, d_laser={d_laser}')
        partcle_params[:, 3][i] = scipy.stats.norm(z + laser_nois["mean"], laser_nois["std"]).pdf(d_laser)

    avg = np.mean(partcle_params[:, 3])
    summ = np.sum(partcle_params[:, 3])
    partcle_params[:, 3] = partcle_params[:, 3] / summ
    sum2 = np.sum(partcle_params[:, 3] ** 2)

    return partcle_params, avg, sum2


def col_oor_handler(
    prtcl_weight,
    polygan=polygan,
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    x0=x0,
    y0=y0,
    map_boundry=map_boundry,
    delete=False,
):
    out = 0
    delete_list = []
    x_list = np.arange(x_min, x_max, 0.02)
    y_list = np.arange(y_min, y_max, 0.02)
    for i in range(prtcl_weight.shape[0]):

        if map.check_is_collition(
            [prtcl_weight[:, 0][i], prtcl_weight[:, 1][i]], polygan
        ) or map.out_of_range(
            [prtcl_weight[:, 0][i], prtcl_weight[:, 1][i]], [x0, y0], map_boundry
        ):
            out = out + 1
            if delete:
                delete_list.append(i)

            else:

                while map.check_is_collition(
                    [prtcl_weight[:, 0][i], prtcl_weight[:, 1][i]], polygan
                ) or map.out_of_range(
                    [prtcl_weight[:, 0][i], prtcl_weight[:, 1][i]],
                    [x0, y0],
                    map_boundry,
                ):
                    prtcl_weight[:, 0][i] = np.round(np.random.choice(x_list) + x0, 2)
                    prtcl_weight[:, 1][i] = np.round(np.random.choice(y_list) + y0, 2)

    if delete:
        prtcl_weight = np.delete(prtcl_weight, delete_list, axis=0)
    summ = prtcl_weight[:, 3].sum()
    prtcl_weight[:, 3] = prtcl_weight[:, 3] / summ
    return prtcl_weight, out


def gen_prtcl(
    x_min=x_min,
    x_max=x_max,
    x0=x0,
    y_min=y_min,
    y_max=y_max,
    y0=y0,
    rotation_list=rotation_list,
    num_prtcls=num_prtcls,
    w=0,
):
    x_list = np.arange(x_min, x_max, 0.01)
    y_list = np.arange(y_min, y_max, 0.01)
    prtcl_x = (np.random.choice(x_list, num_prtcls) + x0).reshape(-1, 1)
    prtcl_y = (np.random.choice(y_list, num_prtcls) + y0).reshape(-1, 1)
    prtcl_theta = (np.random.choice(rotation_list, num_prtcls)).reshape(-1, 1)
    weights = (np.ones(num_prtcls) / num_prtcls).reshape(-1, 1)
    prtcl_weight = np.concatenate([prtcl_x, prtcl_y, prtcl_theta, weights], axis=1)
    return prtcl_weight


def plotter(prtcl_weight, map=map, all_map_lines=all_map_lines):
    # plt.figure()
    plt.clf()
    plt.gca().invert_yaxis()
    map.plot_map(all_map_lines)
    valid_prtcl_sz = prtcl_weight.shape[0]
    xy_prtcl = prtcl_weight[:, :2]
    xy_real = np.array([x_real, y_real]).reshape(1, -1)
    model = DBSCAN(eps=0.03, min_samples=int(0.1 * valid_prtcl_sz))
    yhat = model.fit_predict(xy_prtcl)
    clusters = np.unique(yhat)
    cluster_num = [0]
    for cluster in clusters:
        row_ix = np.where(yhat == cluster)
        if cluster != -1:
            cluster_num.append(len(row_ix[0]) / valid_prtcl_sz)
            if (len(row_ix[0]) / valid_prtcl_sz) > 0.5:
                print(" best estimate : ", np.mean(prtcl_weight[:, :3][row_ix], axis=0))
        plt.scatter(xy_prtcl[row_ix, 1], xy_prtcl[row_ix, 0])

    valid_intsc = False
    max_laser = 0.04
    index = np.argsort(-prtcl_weight[:, 3])[: int(prtcl_weight.shape[0] * 1)]
    prtcl_weight = prtcl_weight[:, :][index]

    max_weight = np.max(prtcl_weight[:, 3])
    norm_weights = prtcl_weight[:, 3] / max_weight

    for i in range(prtcl_weight.shape[0]):
        x = prtcl_weight[i, 0]
        y = prtcl_weight[i, 1]
        theta = prtcl_weight[i, 2]
        weight = norm_weights[i]
        
        # Plot particle as a blue circle
        plt.plot(y, x, 'bo', alpha=weight)
        
        # Plot line indicating laser range
        end_x = x + max_laser * cos(theta)
        end_y = y + max_laser * sin(theta)
        plt.plot([y, end_y], [x, end_x], 'b-', alpha=weight)

    # plt.draw()
    plt.pause(0.1)
    return max(cluster_num) * 100


# def plotter(prtcl_weight , map = map , all_map_lines = all_map_lines  ):
#     plt.figure()
#     plt.clf()
#     plt.gca().invert_yaxis()
#     map.plot_map()

#     # Normalize weights for plotting
#     max_weight = np.max(prtcl_weight[:, 3])
#     norm_weights = prtcl_weight[:, 3] / max_weight

#     for i in range(prtcl_weight.shape[0]):
#         x = prtcl_weight[i, 0]
#         y = prtcl_weight[i, 1]
#         theta = prtcl_weight[i, 2]
#         weight = norm_weights[i]

#         # Plot particle as a blue circle
#         plt.plot(y, x, 'bo', alpha=weight)

#         # Plot line indicating laser range
#         end_x = x + max_laser * cos(theta)
#         end_y = y + max_laser * sin(theta)
#         plt.plot([y, end_y], [x, end_x], 'b-', alpha=weight)

#     plt.draw()
#     plt.pause(0.1)


##### Main Program ####################

rospy.init_node("Particle_Filter_Localization")
velocity_publisher = rospy.Publisher("/motors/wheels", Drive, queue_size=1)
laser_reader = rospy.Subscriber("/proximity", Proximity, callback_laser, queue_size=1)


# initial particles from a uniform distribution:
X_W = (
    gen_prtcl()
)  # generate particle from a uniform distribution with normalized weights

X_W, num_out = col_oor_handler(X_W)

X_W_update = np.copy(X_W)
q = 0

list_portion = []

alpha_slow = 0.3
alpha_fast = 0.7
w_slow = 1e-10
w_fast = 1e-10
w_avg = 1e-10
portion = 0
g = 0
while not rospy.is_shutdown():
    q = q + 1
    print("**************")
    print(" iteration  : " + str(q))

    print("move ........")

    key = input()

    if key in ["w", "s"]:
        angle = 0
        if key == "w":
            dist = 0.1
        elif key == "s":
            dist = -0.1

    elif key in ["a", "d"]:
        dist = 0
        if key == "a":
            angle = PI / 2

        elif key == "d":
            angle = -PI / 2
    elif key == 'q':
        break

    t0 = rospy.Time.now().to_sec()
    while t0 <= 0:
        t0 = rospy.Time.now().to_sec()

    dt = 0
    speed_t = dist / desired_time
    velocity_x = speed_t
    speed_r = angle / desired_time
    velocity_z = speed_r
    laser_reading = laser
    laser_reading1 = laser
    update = False
    move = False

    if dist != 0:
        v = speed_t * 1000
        velocity_publisher.publish(v, v, 0, 0)
        speed_r = 0
    if angle != 0:
        v = speed_r * (0.048 / ((0.176 / PI) ** 0.5)) / 4
        v = v * 1000
        velocity_publisher.publish(-v, v, 0, 0)
        speed_t = 0

    time.sleep(desired_time)
    dt = desired_time
    laser_reading = laser
    # print(laser_reading)

    velocity_publisher.publish(0.0, 0.0, 0.0, 0.0)
    update = True

    if update == True:
        print("laser reading : ", laser_reading)

        # update particles : x(t) = p (x(t-1) , u(t))

        X_W = motion_model(X_W, v=speed_t, w=speed_r, dt=dt)

        X_W, num_out = col_oor_handler(
            X_W, delete=True
        )  ## assigning zero weight to collison and out of range (x y theta)s

        laser_reading = laser

        x_real, y_real, theta_real = x, y, theta

        X_W, w_avg, sum2 = measurment_model(X_W, z=laser_reading)  # Weight calculation

        X_W_update = np.copy(X_W)

        w_slow = w_slow + alpha_slow * (w_avg - w_slow)
        w_fast = w_fast + alpha_fast * (w_avg - w_fast)

        print(
            "w_avg : "
            + str(w_avg)
            + " ,w_slow : "
            + str(w_slow)
            + " ,w_fast : "
            + str(w_fast)
        )
        print(f'w_slow {w_slow}, w_fast {w_fast}, num_prtcls: {num_prtcls}')
        rand_sz = int((np.max([0, 1 - (w_fast / w_slow)])) * num_prtcls)
        rand_sz = np.min([int(0.01 * num_prtcls), rand_sz])

        print("random size : ", rand_sz)

        valid_prtcl_sz = X_W.shape[0]

        k1 = 0.8
        k2 = 0.15
        k3 = 0.05

        X_W_random = gen_prtcl(num_prtcls=int(k3 * num_prtcls + rand_sz))
        X_W_random, num_out = col_oor_handler(X_W_random)

        if valid_prtcl_sz > 0:
            index_best = np.random.choice(
                valid_prtcl_sz, int(num_prtcls * k1), p=X_W[:, 3]
            )
            X_W_best1 = X_W[:, :][index_best]
            index_best = (-X_W[:, 3]).argsort()[: int(k2 * num_prtcls)]
            X_W_best2 = X_W[:, :][index_best]

        else:
            X_W_best1 = gen_prtcl(num_prtcls=int(num_prtcls * k1))
            X_W_best1, num_out = col_oor_handler(X_W_best1)
            X_W_best2 = gen_prtcl(num_prtcls=int(num_prtcls * k2))
            X_W_best2, num_out = col_oor_handler(X_W_best2)

        X_W = np.concatenate([X_W_best1, X_W_best2, X_W_random], axis=0)

    portion = plotter(X_W_update)
    list_portion.append(portion)
    xw_sz = X_W_update.shape[0]
    index_best = (-X_W_update[:, 3]).argsort()[: int(0.8 * xw_sz)]
    X_W_top_estimation = X_W_update[:, :][index_best]
    mno = X_W_top_estimation[0]
    print(" portion 0f close particles % : ", portion)
    print(" best particle : ", mno)
    print(" number of particles : ", X_W.shape[0])
    print("############")

    if portion > 80:
        g = g + 1

    if portion > 80 and g == 1:
        plt.figure()
        plt.plot(list_portion)
        plt.title("Portion of Close Particles")
        plt.show()
    elif portion > 80 and g >= 6:
        plt.figure()
        plt.plot(list_portion)
        plt.title("Portion of Close Particles - kidnapping")
        plt.show()
        break
