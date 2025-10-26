#!/usr/bin/env python3
# Triangle-> Status: FERTILIZER_REQUIRED
# Square-> Status: BAD_HEALTH
# Pentagon(at 1st wp)-> Status: DOCK_STATION

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from math import atan, cos, sin, radians, degrees, pi
import numpy as np
from numpy.random import default_rng
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import time

previous_shape = type('previous', (object,), {'shape_': None, 'x_': None, 'y_': None})()
class ShapeDetector(Node):
    def __init__(self):
        super().__init__('shape_det')
        self.shape_pub=self.create_publisher(String, '/detection_status', 10)    #Status,x,y
        self.odom_sub=self.create_subscription(Odometry,'/odom',self.odom_callback,10)
        self.scan_sub=self.create_subscription(LaserScan,'/scan',self.scan_callback,10)
        
        self.shape_status_map = {
            "triangle": "FERTILIZER_REQUIRED",
            "square": "BAD_HEALTH",
            "pentagon": "DOCK_STATION"
        }
        self.previous_ = previous_shape
        self.DET_BOX_size=1.2  #meters

        self.position_x= -1.5339
        self.position_y= -6.6156
        self.orientation_yaw= 1.57
        # self.timer=self.create_timer(1.0,self.scan_callback)

    def scan_callback(self, msg: LaserScan):
        pts = []
        angle = msg.angle_min
        if(abs(self.position_x+1.5339)<=0.8 and abs(self.position_y+6.6156)<=0.8):
            self.get_logger().info("Skipping detection at starting point.")
            return  #skip detection at starting point to avoid false detection of walls as shapes
        fov_min = radians(0)             # FOV in radians (±135°)
        fov_max = radians(135)

        for r in msg.ranges:
            if msg.range_min < r < self.DET_BOX_size:
                if fov_min <= angle <= fov_max:
                    x = r * cos(angle)
                    y = r * sin(angle)
                    pts.append((x, y))
            angle += msg.angle_increment

        # Convert to numpy array
        if len(pts) > 0:
            point_array = np.array(pts)
        else:
            point_array = np.empty((0, 2))

        if point_array.shape[0] > 0:        # wall filtering (adjust threshold if needed)
            point_array = point_array[np.linalg.norm(point_array, axis=1) > 0.05]
        if point_array.shape[0] > 6:
            
            detected_shape,local_xy = self.detect_shape(point_array)
            if detected_shape != "unknown":
                x_world, y_world = self.body_to_world(local_xy)
                if self.previous_.shape_==None or (abs(x_world-self.previous_.x_)>=0.45 and abs(y_world-self.previous_.y_)>=0.45 and detected_shape!=self.previous_.shape_):
                    status_msg = String()
                    status_msg.data = self.get_status(detected_shape, x_world, y_world)
                    self.shape_pub.publish(status_msg)
                    self.get_logger().info(f"Detected {detected_shape} at world coords ({x_world:.2f}, {y_world:.2f})")
                    self.previous_.shape_ = detected_shape
                    self.previous_.x_ = x_world
                    self.previous_.y_ = y_world
                
            
            # else:
                # self.get_logger().info("Shape not recognized.")
        else:
            self.get_logger().info("Not enough points for shape detection.")

    def odom_callback(self, msg: Odometry):
        self.position_x=msg.pose.pose.position.x
        self.position_y=msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _,_,self.orientation_yaw=euler_from_quaternion([q.x, q.y, q.z, q.w])
    
    def body_to_world(self, local_xy):
        """To transform point from robot frame to world frame."""
        x_l, y_l = local_xy
        x_w = self.position_x + (x_l * cos(self.orientation_yaw) - y_l * sin(self.orientation_yaw))
        y_w = self.position_y + (x_l * sin(self.orientation_yaw) + y_l * cos(self.orientation_yaw))
        return x_w, y_w
    
    def get_status(self, detected_shape,  x_world, y_world):
        status= self.shape_status_map.get(detected_shape, "UNKNOWN_SHAPE")
        return  f"{status},{x_world:.3f},{y_world:.3f}"
    
    def detect_shape(self, point_array, max_edges=6, parallel_angle_tol=5):
        # Detect shape (triangle, square, pentagon) from 2D lidar scan using RANSAC line fits.
        #Assumes one side is hidden by a wall, so visible edges = n-1.
        X = point_array[:, 0].reshape(-1, 1)
        y = point_array[:, 1].reshape(-1, 1)
        slopes=[]
        line_xys=[]
        remaining_points = point_array.copy()
        ransac_results = []

        for i in range(max_edges):
            if remaining_points.shape[0] < 6:
                break
            # iterative edge fitting
            reg = RANSAC(model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error)
            reg.fit(remaining_points[:, 0].reshape(-1, 1), remaining_points[:, 1].reshape(-1, 1))
            if reg.best_fit is None:
                # No valid line found, stop iteration
                break
            slope = reg.best_fit.params[1, 0]   # [intercept, slope]
            intercept = reg.best_fit.params[0, 0]
            ransac_results.append({
                'slope': slope,
                'intercept': intercept
            })

            angle_deg= degrees(atan(slope))
            if len(slopes) > 0:
                prev_angle = slopes[-1]
                if abs(angle_deg - prev_angle) < parallel_angle_tol:
                    # Nearly parallel → retain the last one, skip this
                    continue

            slopes.append(angle_deg)
        # store line point (for coordinate extraction)
            x_mean = np.mean(remaining_points[:, 0])
            y_mean = slope * x_mean + intercept
            line_xys.append((x_mean, y_mean))

            # inliers removed from remaining points
            y_pred = reg.predict(remaining_points[:, 0].reshape(-1, 1))
            inliers_mask = ((remaining_points[:, 1].reshape(-1, 1) - y_pred) ** 2 < reg.t).flatten()
            remaining_points = remaining_points[~inliers_mask]
        
        visible_edges = len(slopes)
        if visible_edges<2:
            return "unknown", (0.0, 0.0)
        
        # Compute angle differences between consecutive edges
        inter_edge_angles = []
        for i in range(visible_edges - 1):
            m1 = np.tan(np.radians(slopes[i]))
            m2 = np.tan(np.radians(slopes[i + 1]))
            theta = abs(atan((m2 - m1) / (1 + m1 * m2)))
            inter_edge_angles.append(degrees(theta))

        # --- Classification rules ---
        if visible_edges == 2:
            # Two edges meeting at acute or right angle → triangle
            if any(80 < a < 100 for a in inter_edge_angles):
                detected_shape = "square"
            elif any(30 < a < 88 for a in inter_edge_angles):
                detected_shape = "triangle"
            else:
                detected_shape = "unknown"

        elif visible_edges == 3:
            # Three edges meeting ~90° → square
            ninety_count = sum(abs(a - 90) < 10 for a in inter_edge_angles)
            detected_shape = "square" if ninety_count >= 2 else "unknown"

        elif visible_edges <= 4:
            # Four visible edges → likely pentagon
            if any(100 <= a <= 120 for a in inter_edge_angles):
                detected_shape = "pentagon"
            else:
                detected_shape = "unknown"

        else:
            detected_shape = "unknown"

        if len(line_xys) > 0:
            local_shape_point = tuple(
                map(float, reversed([sum(coords) / len(line_xys) for coords in zip(*line_xys)]))
            )
        else:
            local_shape_point = (0.0, 0.0)
        if detected_shape != "unknown":
            # self.get_logger().info(f"Shape={detected_shape}, edges={visible_edges}, local_point={np.round(local_shape_point,3)}, angles={np.round(inter_edge_angles,1)}degrees")
            self.visualize_ransac_edges(point_array, ransac_results)
        return detected_shape, local_shape_point


    def visualize_ransac_edges(self,scan_points, ransac_results, title="RANSAC Detected Edges"):
        plt.figure(figsize=(8, 6))
        plt.scatter(scan_points[:, 0], scan_points[:, 1], s=5, color='gray', label='LiDAR points')

        colors = plt.cm.get_cmap('tab10', len(ransac_results))

        for i, result in enumerate(ransac_results):
            slope = result.get('slope')
            intercept = result.get('intercept')

            if slope is None or intercept is None:
                continue

            # Line visualization range
            x_vals = np.linspace(np.min(scan_points[:, 0]), np.max(scan_points[:, 0]), 200)
            y_vals = slope * x_vals + intercept

            plt.plot(x_vals, y_vals, color=colors(i), linewidth=1, label=f'Edge {i+1}')


        plt.legend()
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.show(block=False)  # non-blocking show
        plt.pause(0.2)         # display for 0.2 seconds
        plt.close()            # close figure


rng = default_rng()

class RANSAC:
    def __init__(self, n=4, k=500, t=0.01, d=8, model=None, loss=None, metric=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.loss = loss        # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric    # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf

    def fit(self, X, y):
        for _ in range(self.k):
            ids = rng.permutation(X.shape[0])

            maybe_inliers = ids[: self.n]
            maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])

            thresholded = (
                self.loss(y[ids][self.n :], maybe_model.predict(X[ids][self.n :]))
                < self.t
            )

            inlier_ids = ids[self.n :][np.flatnonzero(thresholded).flatten()]

            if inlier_ids.size > self.d:
                inlier_points = np.hstack([maybe_inliers, inlier_ids])
                x_span = X[inlier_points, 0].max() - X[inlier_points, 0].min()
                # reject if the line is too long
                if x_span > 0.35:
                    continue
                better_model = copy(self.model).fit(X[inlier_points], y[inlier_points])

                this_error = self.metric(
                    y[inlier_points], better_model.predict(X[inlier_points])
                )

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = better_model

        return self

    def predict(self, X):
        return self.best_fit.predict(X)

def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]


class LinearRegressor:
    def __init__(self):
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        self.params = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        return X @ self.params

def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown() 

if __name__ == '__main__':
    main()

# if __name__ == "__main__":

#     regressor = RANSAC(model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error)

#     X = np.array([-0.848,-0.800,-0.704,-0.632,-0.488,-0.472,-0.368,-0.336,-0.280,-0.200,-0.00800,-0.0840,0.0240,0.100,0.124,0.148,0.232,0.236,0.324,0.356,0.368,0.440,0.512,0.548,0.660,0.640,0.712,0.752,0.776,0.880,0.920,0.944,-0.108,-0.168,-0.720,-0.784,-0.224,-0.604,-0.740,-0.0440,0.388,-0.0200,0.752,0.416,-0.0800,-0.348,0.988,0.776,0.680,0.880,-0.816,-0.424,-0.932,0.272,-0.556,-0.568,-0.600,-0.716,-0.796,-0.880,-0.972,-0.916,0.816,0.892,0.956,0.980,0.988,0.992,0.00400]).reshape(-1,1)
#     y = np.array([-0.917,-0.833,-0.801,-0.665,-0.605,-0.545,-0.509,-0.433,-0.397,-0.281,-0.205,-0.169,-0.0531,-0.0651,0.0349,0.0829,0.0589,0.175,0.179,0.191,0.259,0.287,0.359,0.395,0.483,0.539,0.543,0.603,0.667,0.679,0.751,0.803,-0.265,-0.341,0.111,-0.113,0.547,0.791,0.551,0.347,0.975,0.943,-0.249,-0.769,-0.625,-0.861,-0.749,-0.945,-0.493,0.163,-0.469,0.0669,0.891,0.623,-0.609,-0.677,-0.721,-0.745,-0.885,-0.897,-0.969,-0.949,0.707,0.783,0.859,0.979,0.811,0.891,-0.137]).reshape(-1,1)

#     regressor.fit(X, y)

#     import matplotlib.pyplot as plt
#     plt.style.use("seaborn-darkgrid")
#     fig, ax = plt.subplots(1, 1)
#     ax.set_box_aspect(1)

#     plt.scatter(X, y)

#     line = np.linspace(-1, 1, num=100).reshape(-1, 1)
#     plt.plot(line, regressor.predict(line), c="peru")
#     plt.show()



        # # Removing inlier points of first line
        # y_pred = regressor1.predict(X)
        # inliers_mask = ((y - y_pred) ** 2 < regressor1.t).flatten()
        # remaining_points = point_array[~inliers_mask]

        # if len(remaining_points) < 10:
        #     print("Only one line detected — likely flat wall.")
        #     return "unknown"

        # # Fit second edge over remaining points
        # X2 = remaining_points[:, 0].reshape(-1, 1)
        # y2 = remaining_points[:, 1].reshape(-1, 1)
        # regressor2 = RANSAC(model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error)
        # regressor2.fit(X2, y2)
        # slope_l2 = regressor2.best_fit.params[1, 0]

        # # Compute angle between the two lines
        # theta = abs(atan((slope_l2 - slope_l1) / (1 + slope_l1 * slope_l2)))
        # theta_deg = degrees(theta)

        # # Use more RANSAC fits if you want 3 or 4 edges (square/pentagon)
        # # For now, we detect up to two visible edges for simplicity
        # print(f"Angle between lines = {theta_deg:.2f}°")

        # # --- Shape classification logic ---
        # if theta_deg < 80:
        #     detected_shape = "triangle"
        # elif abs(theta_deg - 90) < 10:
        #     detected_shape = "square"
        # elif 100 <= theta_deg <= 120:
        #     detected_shape = "pentagon"
        # else:
        #     detected_shape = "unknown"

        # print("Detected shape:", detected_shape)
        # return detected_shape
        # regressor = RANSAC(model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error)
        # regressor.fit(point_array[:,0].reshape(-1,1), point_array[:,1].reshape(-1,1))
        # slope_l1,slope_l2= regressor.best_fit.params[1],regressor.best_fit.params[2]
        # angle_bw_lines=atan((slope_l1-slope_l2)/(1+slope_l1*slope_l2)) #degrees
        # if angle_bw_lines<pi/2.0:#some_condition_for_triangle:
        #     detected_shape = "triangle"
        # elif angle_bw_lines==pi/2.0:
        #     detected_shape = "square"
        # elif some_condition_for_pentagon:
        #     detected_shape = "pentagon"
        # else:
        #     detected_shape = "unknown"
        # print("Detected shape:", detected_shape)  # Debug print for verification
        # return detected_shape  # Example return value