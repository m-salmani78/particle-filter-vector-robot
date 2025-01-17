#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
import numpy as np
from math import *
from os.path import expanduser
import matplotlib.pyplot as plt
import scipy

from map import Map


## Constants and parameters:
num_prticles = 1000

laser_var = 0.0029 * 5
v_var = 0.008 / 2
w_var = 0.09 * 0.5
g_var = (0.01349) / 200

x = 0.0
y = 0.0 
theta = 0.0

PI = 3.1415926535897

laser_data=0
laser=0

desired_time=3

rotation_list = [ PI/2 , 3*(PI/2) , 0 , PI ]

home = expanduser("~")
map_address = home + '/catkin_ws/src/anki_description/world/sample1.world'
map = Map(map_address)
x_min , x_max , y_min , y_max = map.map_boundry

# Noises
translation_model_params = {
    0: {"mean": 0.0, "std": 0.0, "dt":0.0},
    5: {"mean": -0.0084, "std": 0.004, "dt":0.1},
    10: {"mean": -0.0091, "std": 0.004, "dt":0.2},
    15: {"mean": -0.0086, "std": 0.004, "dt":0.3},
}

rotation_model_params = {
    0: {"mean": 0.0, "std": 0.0, "dt":0.0},
    90: {"mean": 0.564, "std": 0.0014, "dt":PI/2},
    -90: {"mean": -0.577, "std": 0.0012, "dt":-PI/2},
}

# Functions:
def callback_laser(msg):
    global laser
    laser= msg.range

def new_odometry(msg: Odometry):
    global x
    global y
    global theta
 
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
 
    rot_q = msg.pose.pose.orientation
    (roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])


def motion_model(prtcl_weight, v, w, dt, v_noise=translation_model_params[0], w_noise=rotation_model_params[0], g_var=g_var):

    for i in range(prtcl_weight.shape[0]):
        v_hat = v + np.random.normal(v_noise["mean"], v_noise["std"])
        w_hat = w + np.random.normal(w_noise["mean"], w_noise["std"])
        if w_hat==0:
            w_hat=1e-9
        g_hat = np.random.normal(0, g_var)

        prtcl_weight[:,0][i] = prtcl_weight[:,0][i] + (v_hat/w_hat)* ( - sin(prtcl_weight[:,2][i]) + sin(prtcl_weight[:,2][i]+ w_hat*dt) ) 

        prtcl_weight[:,1][i] = prtcl_weight[:,1][i] + (v_hat/w_hat)* (   cos(prtcl_weight[:,2][i]) - cos(prtcl_weight[:,2][i]+ w_hat*dt) )

        prtcl_weight[:,2][i] = prtcl_weight[:,2][i]+ dt * (w_hat + g_hat) + 10*PI
        prtcl_weight[:,2][i] = prtcl_weight[:,2][i] - int(prtcl_weight[:,2][i]/(2*PI))*2*PI
   
    return prtcl_weight

def measurment_model (prtcl_weight , z ,laser_var = laser_var ,  map =map) :
    max_laser = 0.435 #35 2
    sensor_var = laser_var
    print(prtcl_weight.shape)
    
    for i in range(prtcl_weight.shape[0]):

        laser_available=False

        prtcl_start = [ prtcl_weight[:,0][i] , prtcl_weight[:,1][i] ]  
        prtcl_end =   [ prtcl_weight[:,0][i] + max_laser*cos(prtcl_weight[:,2][i])  , prtcl_weight[:,1][i] + max_laser*sin(prtcl_weight[:,2][i])]
        min_distance=10
        col = False
        for line in map.all_map_lines:
            intersection_point = map.find_intersection(line[0], line[1] , prtcl_start, prtcl_end)
            if intersection_point != False:
                d_laser = ((intersection_point[0]-prtcl_weight[:,0][i] )**2 + (intersection_point[1]-prtcl_weight[:,1][i] )**2  )**0.5
                if min_distance >= d_laser:
                    min_distance=d_laser
                    col=intersection_point
            
        if not col:
            min_distance=max_laser
        
        d_laser = min_distance


        prtcl_weight[:,3][i] = scipy.stats.norm(d_laser , sensor_var).pdf(z+0.035)
    
    avg = np.mean( prtcl_weight[:,3] )
    summ = np.sum(prtcl_weight[:,3])
    prtcl_weight[:,3] = prtcl_weight[:,3] / summ
    sum2 = np.sum(prtcl_weight[:,3]**2)
    
    return prtcl_weight , avg , sum2



def col_oor_handler (prtcl_weight, x_min=x_min , x_max=x_max , y_min = y_min , y_max= y_max, map=map , delete= False):
    out=0
    delete_list=[]
    x_list = np.arange(x_min , x_max , 0.02)
    y_list = np.arange(y_min , y_max , 0.02)
    for i in range(prtcl_weight.shape[0]):

        if map.check_is_collition([ prtcl_weight[:,0][i] , prtcl_weight[:,1][i] ]) or map.out_of_range([ prtcl_weight[:,0][i] , prtcl_weight[:,1][i] ],[map.x0 , map.y0]):
            out = out+1
            if delete :
                delete_list.append(i)

            else:

                while map.check_is_collition([ prtcl_weight[:,0][i] , prtcl_weight[:,1][i] ]) or map.out_of_range([ prtcl_weight[:,0][i] , prtcl_weight[:,1][i] ],[map.x0 , map.y0]):
                    prtcl_weight[:,0][i] = np.round(np.random.choice(x_list) + map.x0,2)
                    prtcl_weight[:,1][i]  = np.round(np.random.choice(y_list)+ map.y0 , 2)

    
    if delete:
        prtcl_weight=np.delete(prtcl_weight , delete_list , axis=0 )
    summ = prtcl_weight[:,3].sum()
    prtcl_weight[:,3] = prtcl_weight[:,3] / summ
    return prtcl_weight , out



def generate_prticles(x_min=x_min , x_max=x_max , x0=map.x0 , y_min=y_min , y_max=y_max , y0=map.y0 , rotation_list=rotation_list,  num_prtcls=num_prticles , w=0):
    x_list = np.arange(x_min , x_max , 0.02)
    y_list = np.arange(y_min , y_max , 0.02)
    prtcl_x = (np.random.choice(x_list , num_prtcls) + x0).reshape(-1,1) 
    prtcl_y = (np.random.choice(y_list, num_prtcls) + y0).reshape(-1,1)
    prtcl_theta = (np.random.choice(rotation_list, num_prtcls)).reshape(-1,1)
    weights = (np.ones(num_prtcls)/num_prtcls).reshape(-1,1)
    prtcl_weight = np.concatenate([prtcl_x,prtcl_y,prtcl_theta , weights],axis=1)
    return prtcl_weight 

def plotter(prtcl_weight, x_val, y_val, theta_val, map=map, max_laser=0.04):
    plt.clf()
    plt.gca().invert_yaxis()
    map.plot_map()

    # True position of the robot :
    plt.arrow(y_val, x_val, 1e-5*sin(theta_val), 1e-5*cos(theta_val), color='yellow', head_width=0.02, overhang=0.6)
    circle1 = plt.Circle((y_val, x_val), 0.05, color='r')
    plt.gca().add_patch(circle1)
    
    # Normalize weights for plotting
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
    
    plt.draw()
    plt.pause(0.1)


##### Main Program ####################

rospy.init_node("Particle_Filter_Localization")
odometry_reader= rospy.Subscriber("/odom", Odometry, new_odometry)
velocity_publisher = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=1)
laser_reader = rospy.Subscriber('/vector/laser', Range, callback_laser, queue_size = 1)
vel_msg = Twist()

# initial particle generation
X_W = generate_prticles()

X_W , num_out = col_oor_handler(X_W)

X_W_update=np.copy(X_W)
q=0
x_real , y_real , theta_real = x , y , theta
list_portion=[]

alpha_slow = 0.3
alpha_fast = 0.7
w_slow=0
w_fast=0
w_avg = 0
portion = 0
while not rospy.is_shutdown():
    q=q+1
    print("**************")
    print ( " iteration  : " + str(q))

    print("move ........")

    key=input()

    if key in ['w','ww','www']:
        angle = rotation_model_params[0]
        if key == 'w' :
            distance = translation_model_params[5]
        elif key == 'ww':
            distance = translation_model_params[10]
        elif key == 'www':
            distance = translation_model_params[15]
        # elif key == 's':
        #     distance = -0.1
        # elif key == 'ss':
        #     distance = -0.2
    
    elif key in ['a', 'd']:
        distance = translation_model_params[0]
        if key=='a':
            angle= rotation_model_params[90]
        # elif key == 'aa' :
        #     angle = PI   
        elif key == 'd' :
            angle = rotation_model_params[-90]
        # elif key == 'dd' :
        #     angle = -PI
    
    elif key == 'q':
        break


    t0 = rospy.Time.now().to_sec()
    while t0<=0:
        t0 = rospy.Time.now().to_sec()

    dt=0
    speed_t = (distance["dt"] /desired_time)
    velocity_x = speed_t
    speed_r=(angle["dt"] /desired_time)
    velocity_z = speed_r
    laser_reading = laser
    laser_reading1 = laser
    update=False
    move=False
    while(dt<desired_time):
        if (laser_reading>0.04) :
            vel_msg.linear.x = velocity_x
            vel_msg.angular.z=velocity_z *2 ## double velocity due to the simulation bug
            velocity_publisher.publish(vel_msg)
            t1=rospy.Time.now().to_sec()
            dt=t1-t0
            update=True
            move=True
        elif (laser_reading<=0.04) and (angle["dt"]!=0):
            vel_msg.linear.x = velocity_x
            vel_msg.angular.z=velocity_z *2 ## double velocity due to the simulation bug
            velocity_publisher.publish(vel_msg)
            t1=rospy.Time.now().to_sec()
            dt=t1-t0
            update=True
            move=True   
        else:
            break
        laser_reading = laser

    laser_reading = laser
    vel_msg.linear.x = 0
    vel_msg.angular.z = 0
    velocity_publisher.publish(vel_msg)
    #update=True
    
    if update == True:
        

        # update particles : x(t) = p (x(t-1) , u(t))

        X_W = motion_model(X_W , v = speed_t , w = speed_r , dt = dt, v_noise=distance, w_noise=angle)

        X_W , num_out = col_oor_handler(X_W , delete=True) ## assigning zero weight to collison and out of range (x y theta)s


        laser_reading = laser

        x_real , y_real , theta_real = x , y , theta

        X_W , w_avg  , sum2 = measurment_model(X_W , z=laser_reading ) # Weight calculation

        X_W_update=np.copy(X_W)

        xy_prtcl = X_W[:,:2]
        xy_real = np.array([x_real , y_real ]).reshape(1,-1)

        close_prtcls_index = np.where( np.linalg.norm( xy_prtcl - xy_real , axis=1) <=0.05 )
        portion = ( (len(close_prtcls_index[0]) / X_W_update.shape[0] )*100) 
        list_portion.append(portion)
        print( " Portion of close particles = " + str ( portion) + " % ")


        w_slow = w_slow + alpha_slow * (w_avg-w_slow)
        w_fast = w_fast + alpha_fast * (w_avg-w_fast)

        print("w_avg : " + str (w_avg ) + " ,w_slow : " + str(w_slow) + " ,w_fast : " + str(w_fast) ) 
        rand_sz = int((np.max( [ 0 , 1-(w_fast/w_slow)]) ) * num_prticles )
        rand_sz = np.min( [int(0.1* num_prticles) , rand_sz] )

        print("random size : " , rand_sz)

        valid_prtcl_sz = X_W.shape[0]

        k1=0.7
        k2 = 0.25
        k3=0.05
        
        X_W_random = generate_prticles(num_prtcls=int(k3*num_prticles + rand_sz))
        X_W_random , num_out= col_oor_handler(X_W_random)

        if valid_prtcl_sz > 0 :
            index_best = np.random.choice (valid_prtcl_sz,int(num_prticles*k1), p= X_W[:,3])
            X_W_best1 = X_W[:,:][index_best]
            index_best = (-X_W[:,3]).argsort()[:int(k2 * num_prticles)]
            X_W_best2 = X_W[:,:][index_best]
                        
        else:
            X_W_best1= generate_prticles(num_prtcls=int(num_prticles*k1))
            X_W_best1 , num_out= col_oor_handler(X_W_best1)
            X_W_best2= generate_prticles(num_prtcls=int(num_prticles*k2))
            X_W_best2 , num_out= col_oor_handler(X_W_best2)
                    

        X_W = np.concatenate( [ X_W_best1 , X_W_best2 , X_W_random ] , axis=0 )
        


    plotter( X_W_update , x_real , y_real , theta_real )
    xw_sz = X_W_update.shape[0]
    index_best = (-X_W_update[:,3]).argsort()[:int(0.8 * xw_sz)]
    X_W_top_estimation = X_W_update[:,:][index_best]
    mno = X_W_top_estimation
    X_W_top_estimation = X_W_top_estimation[:,:3]
    X_W_mean_estimation = np.mean(X_W_top_estimation , axis=0)


    print(" best particle : " , mno[0])
    
    print( " real pose : " ,x , y , theta)
    print( " estimated pose : " , X_W_mean_estimation )
    print( " number of particles : " , X_W.shape[0])
    print("############")

    
    
    g = 0
    if portion > 80:
        plt.figure()
        plt.plot(list_portion)
        plt.title( "Portion of Close Particles")
        plt.show()
        g=1
    elif portion > 80 and g==1 :
        plt.figure()
        plt.plot(list_portion)
        plt.title( "Portion of Close Particles - kidnapping")
        plt.show() 
        break