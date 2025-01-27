import rclpy
import cv2  
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class FaceDetection(Node):
    def __init__(self):
        super().__init__('face_detection')

        # Publisher for webcam images
        self.pub_camera = self.create_publisher(Image, '/webcam', 10)
        
        # Initialize VideoCapture to read from webcam
        self.image = cv2.VideoCapture(0)
        
        if not self.image.isOpened():
            self.get_logger().error("Failed to open the webcam.")
            return

        # Path to your face detection classifier
        face_cascade_path = '/home/aniket-subudhi/face_detection_ws/src/Integrating-Open-cv-with-ROS-2/ros2_cv/ros2_cv/facedetection.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

        if self.face_cascade.empty():
            self.get_logger().error(f"Failed to load face detection XML file: {face_cascade_path}")
            return

        # Initialize CvBridge to convert OpenCV images to ROS Image messages
        self.bridge = CvBridge()

    def show(self):
        while rclpy.ok():
            ret, frame = self.image.read()
            if not ret:
                self.get_logger().error("Failed to read from the webcam.")
                break

            # Detect faces in the frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the frame with face detection
            cv2.imshow("Face Detection", frame)

            # Publish the image as ROS Image message
            try:
                img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.pub_camera.publish(img_msg)
            except CvBridgeError as e:
                self.get_logger().error(f"Failed to convert frame to ROS Image message: {e}")

            # Exit if 'q' is pressed
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        self.image.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    
    face_detection_node = FaceDetection()

    # Only run the node if initialization was successful
    if face_detection_node.image.isOpened():
        face_detection_node.show()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
