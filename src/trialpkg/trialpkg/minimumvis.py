import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

import sys
sys.path.append('/home/atharva/yolov5')
from models.experimental import attempt_load
from utils.general import non_max_suppression

class WorkpieceVisionNode(Node):
    def __init__(self):
        super().__init__('workpiece_vision_node')
        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.rgb_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.rgb_info_callback, 10)
        self.depth_info_sub = self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.depth_info_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/workpiece_pose', 10)
        self.yolo_img_pub = self.create_publisher(Image, '/yolo/detections/image', 10)

        self.rgb_image = None
        self.depth_image = None
        self.rgb_info = None
        self.depth_info = None

        # Load YOLOv5
        weights = '/home/atharva/yolov5/runs/train/exp3/weights/best.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load(weights, device=self.device)
        self.model.eval()
        self.get_logger().info("YOLOv5 loaded.")

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    def rgb_info_callback(self, msg):
        k = msg.k
        self.fx, self.fy = k[0], k[4]
        self.cx, self.cy = k[2], k[5]
        self.rgb_info = msg

    def depth_info_callback(self, msg):
        self.depth_info = msg
    
    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.try_process()

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.try_process()

    def try_process(self):
        if any(x is None for x in (self.rgb_image, self.depth_image, self.rgb_info, self.depth_info)):
            return
        self.detect_and_publish()

    def detect_and_publish(self):
        img = self.rgb_image
        img_yolo = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).to(self.device).permute(2,0,1).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]
        boxes = pred.cpu().numpy() if pred is not None else []
        if len(boxes) == 0:
            self.get_logger().info("No objects detected.")
            return

        h_orig, w_orig = img.shape[:2]
        scale_x, scale_y = w_orig / 640, h_orig / 640

        xmin, ymin, xmax, ymax, conf, cls = boxes[0].astype(int)
        xmin = int(xmin * scale_x)
        xmax = int(xmax * scale_x)
        ymin = int(ymin * scale_y)
        ymax = int(ymax * scale_y)

        img_annot = img.copy()
        cv2.rectangle(img_annot, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        cv2.putText(img_annot, f'YOLO {conf:.2f}', (xmin, ymin-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        roi = img[ymin:ymax, xmin:xmax]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().warn("No contours found.")
            self.publish_yolo_image(img_annot)
            return

        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) != 4:
            self.get_logger().warn("Did not find four corners.")
            self.publish_yolo_image(img_annot)
            return

        # Find 3D coordinates of all four corners
        pts = approx.reshape(4,2)
        pts_global = [(int(px + xmin), int(py + ymin)) for px, py in pts]
        
        # Draw corners on the annotated image
        for idx, (cx, cy) in enumerate(pts_global):
            cv2.circle(img_annot, (cx, cy), 6, (0,0,255), -1)
            cv2.putText(img_annot, str(idx+1), (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # Publish the annotated image
        self.publish_yolo_image(img_annot)

        pts_arr = np.array(pts_global)
        s = pts_arr.sum(axis=1)
        bottom_left_idx = np.lexsort((pts_arr[:,0], -pts_arr[:, 1]))[0]
        bottom_left = pts_global[bottom_left_idx]

        # Get depth at that point
        def get_depth_at(x, y):
            roi = self.depth_image[max(y-2,0):y+3, max(x-2,0):x+3]
            valid = roi[np.isfinite(roi) & (roi>0)]
            return float(np.median(valid)) if valid.size>0 else None

        x_bl, y_bl = bottom_left
        d = get_depth_at(x_bl, y_bl)
        if d is None or d <= 0:
            self.get_logger().warn("Bad depth at bottom-left corner.")
            return
        
        # Project to 3D (in camera optical frame)
        x = -d
        y = ((self.cy - y_bl) * d / self.fy)
        z = (x_bl - self.cx) * d / self.fx
        
        # Orientation: basic, or you can compute from corners as in your previous node
        pose = PoseStamped()
        pose.header.frame_id = 'd435_camera_optical_frame'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0  # No orientation calculated here, set as needed

        self.pose_pub.publish(pose)

    def publish_yolo_image(self, img_annot):
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img_annot, encoding='bgr8')
            self.yolo_img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish annotated YOLO image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = WorkpieceVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
