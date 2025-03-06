### ITO YUNG GUMAGANA NA

# import cv2
# import mediapipe as mp
# import numpy as np
# from deepface import DeepFace

# # ----------------------------
# # Initialize MediaPipe Pose and DeepFace
# # ----------------------------
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# pose = mp_pose.Pose(
#     static_image_mode=True,
#     model_complexity=2,
#     enable_segmentation=False
# )

# DeepFace.build_model('VGG-Face')

# # ----------------------------
# # Calibration Constants (Reference)
# # ----------------------------
# REFERENCE_HEIGHT = 170
# REFERENCE_DISTANCE_CM = 150     

# # ----------------------------
# # Helper Functions
# # ----------------------------
# def get_head_and_feet(landmarks):
#     """Extract head (nose) and averaged feet (from left and right foot indices) positions."""
#     head = landmarks[mp_pose.PoseLandmark.NOSE]
#     left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
#     right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
#     foot = type("Point", (object,), {})()  # create a simple object to hold averaged values
#     foot.x = (left_foot.x + right_foot.x) / 2
#     foot.y = (left_foot.y + right_foot.y) / 2
#     foot.z = (left_foot.z + right_foot.z) / 2
#     return head, foot

# def compute_norm_height(head, foot):
#     """Compute the Euclidean distance (normalized) between head and feet landmarks."""
#     head_arr = np.array([head.x, head.y, head.z])
#     foot_arr = np.array([foot.x, foot.y, foot.z])
#     return np.linalg.norm(foot_arr - head_arr)

# def compute_pixel_height(head, foot, image_width, image_height):
#     """Compute the pixel distance between head and foot, converting normalized coords to pixels."""
#     head_px = np.array([head.x * image_width, head.y * image_height])
#     foot_px = np.array([foot.x * image_width, foot.y * image_height])
#     return np.linalg.norm(foot_px - head_px)

# # ----------------------------
# # Step 1: Calibrate Camera from the Reference Image
# # ----------------------------
# def calibrate_reference(reference_image_path):
#     ref_image = cv2.imread(reference_image_path)
#     if ref_image is None:
#         raise IOError("Error: Could not load reference image.")
#     ref_h, ref_w = ref_image.shape[:2]
#     ref_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
#     results = pose.process(ref_rgb)
#     if not results.pose_landmarks:
#         raise ValueError("No pose detected in reference image.")
    
#     landmarks = results.pose_landmarks.landmark
#     head, foot = get_head_and_feet(landmarks)
    
#     # Normalized head-to-foot distance (unitless)
#     norm_height = compute_norm_height(head, foot)
#     # Pixel height from the reference image
#     pixel_height = compute_pixel_height(head, foot, ref_w, ref_h)
    
#     # Compute effective focal length using the pinhole camera model:
#     # f = (pixel_height * distance) / real_height
#     focal_length = (pixel_height * REFERENCE_DISTANCE_CM) / REFERENCE_HEIGHT
    
#     # Also compute a scaling factor for normalized units (for reference image)
#     scaling_factor_ref = REFERENCE_HEIGHT / norm_height
    
#     print(f"[Calibration] Reference pixel height: {pixel_height:.2f} pixels, Focal Length: {focal_length:.2f}")
#     return focal_length, scaling_factor_ref, norm_height

# # ----------------------------
# # Step 2: Process New Image and Compute Real Height
# # ----------------------------
# def process_new_image(new_image_path, focal_length, new_distance_cm):
#     new_image = cv2.imread(new_image_path)
#     if new_image is None:
#         raise IOError("Error: Could not load new image.")
#     new_h, new_w = new_image.shape[:2]
    
#     # Optional: Analyze demographics using DeepFace
#     pred = DeepFace.analyze(new_image, actions=['gender', 'race'], enforce_detection=False)
#     dominant_gender = pred[0]['dominant_gender']
#     dominant_race = pred[0]['dominant_race']
    
#     new_rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
#     results = pose.process(new_rgb)
#     if not results.pose_landmarks:
#         raise ValueError("No pose detected in new image.")
    
#     landmarks = results.pose_landmarks.landmark
#     head, foot = get_head_and_feet(landmarks)
    
#     # Normalized model height from new image
#     norm_height_new = compute_norm_height(head, foot)
#     # Pixel height in new image
#     pixel_height_new = compute_pixel_height(head, foot, new_w, new_h)

#     real_height_new = (pixel_height_new * new_distance_cm) / focal_length
    
#     # Derive a new scaling factor to convert all normalized measurements to cm:
#     scaling_factor_new = real_height_new / norm_height_new
    
#     print(f"[New Image] Pixel height: {pixel_height_new:.2f} pixels, Real Height: {real_height_new:.2f} cm")
#     return new_image, landmarks, scaling_factor_new, dominant_gender, dominant_race, real_height_new


# def dist_3d(a, b):
#     return np.linalg.norm(np.array([a.x, a.y, a.z]) - np.array([b.x, b.y, b.z]))

# def circumference_approx(a, b, scaling_factor, factor=1.0):
#     width = dist_3d(a, b) * scaling_factor
#     return np.pi * width * factor

# def compute_measurements(landmarks, scaling_factor):

#     l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
#     r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#     l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
#     r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
#     l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
#     r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
#     l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
#     l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
#     nose = landmarks[mp_pose.PoseLandmark.NOSE]
#     l_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
#     r_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
#     l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
#     r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
#     l_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
#     l_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    
#     foot_center = np.array([
#         (l_foot.x + r_foot.x) / 2,
#         (l_foot.y + r_foot.y) / 2,
#         (l_foot.z + r_foot.z) / 2
#     ])
#     head_position = np.array([nose.x, nose.y, nose.z])
#     model_height = np.linalg.norm(foot_center - head_position) * scaling_factor
    
#     waist_factor = 1.3   
#     chest_factor = 1.0    
#     thigh_factor = 0.7    
#     wrist_factor = 0.2    
#     neck_factor = 0.35  
    
#     waist_circ = circumference_approx(l_hip, r_hip, scaling_factor, factor=waist_factor)
#     belly_circ = waist_circ * 1.1 
#     chest_circ = circumference_approx(l_shoulder, r_shoulder, scaling_factor, factor=chest_factor)
#     thigh_circ = circumference_approx(l_hip, l_knee, scaling_factor, factor=thigh_factor)
#     wrist_circ = circumference_approx(r_wrist, r_elbow, scaling_factor, factor=wrist_factor)
#     neck_circ = circumference_approx(l_shoulder, r_shoulder, scaling_factor, factor=neck_factor)
    
#     upper_arm_length = dist_3d(l_shoulder, l_wrist) * scaling_factor
#     hand_length = dist_3d(l_wrist, l_index) * scaling_factor
#     total_arm_length = upper_arm_length + hand_length
    
#     foot_length = dist_3d(l_heel, l_foot) * scaling_factor
#     adjusted_ankle_length = foot_length
    
#     measurements = {
#         "Estimated Height (cm)": model_height,
#         "Waist Circumference (cm)": waist_circ,
#         "Belly Circumference (cm)": belly_circ,
#         "Chest Circumference (cm)": chest_circ,
#         "Thigh Circumference (cm)": thigh_circ,
#         "Wrist Circumference (cm)": wrist_circ,
#         "Neck Circumference (cm)": neck_circ,
#         "Arm Length (Left) (cm)": total_arm_length,
#         "Ankle/Foot Length (Left) (cm)": adjusted_ankle_length
#     }
#     return measurements

# def main():
#     reference_image_path = "C:/FaceRace/dataset/170cm.jpg"
#     new_image_path = "C:/FaceRace/dataset/166cm.jpg"

#     focal_length, _, ref_norm_height = calibrate_reference(reference_image_path)

#     new_distance_cm = 150  
    
#     # Process the new image
#     (new_image, landmarks, scaling_factor_new, 
#      dominant_gender, dominant_race, real_height_new) = process_new_image(new_image_path, focal_length, new_distance_cm)
    
#     # Compute measurements using the new scaling factor
#     measurements = compute_measurements(landmarks, scaling_factor_new)
    
#     # Compute body fat percentage using an empirical formula
#     if dominant_gender.lower() == "male":
#         bf_percent = 10.14 + (0.52 * measurements["Waist Circumference (cm)"]) - (0.16 * measurements["Estimated Height (cm)"])
#     else:
#         bf_percent = 29.33 + (0.52 * measurements["Waist Circumference (cm)"]) - (0.19 * measurements["Estimated Height (cm)"])
    
#     # Print the results
#     print(f"Dominant Gender: {dominant_gender}")
#     print(f"Dominant Race: {dominant_race}")
#     print(f"Real Height from New Image: {real_height_new:.2f} cm")
#     for key, value in measurements.items():
#         print(f"{key}: {value:.2f}")
#     print(f"Body Fat Percentage: {bf_percent:.2f}%")
    
#     # Visualization: annotate and display image
#     annotated_image = new_image.copy()
#     mp_drawing.draw_landmarks(annotated_image, 
#                               pose.process(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)).pose_landmarks, 
#                               mp_pose.POSE_CONNECTIONS)
#     cv2.putText(annotated_image, f"Height: {measurements['Estimated Height (cm)']:.2f} cm", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     cv2.putText(annotated_image, f"Body Fat: {bf_percent:.1f}%", (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
#     cv2.imshow("Body Measurements", annotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


    
import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN  # Install via: pip install mtcnn
import math

# ----------------------------
# Initialize MediaPipe Pose and DeepFace
# ----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False
)

DeepFace.build_model('VGG-Face')

REFERENCE_HEIGHT = 146
REFERENCE_DISTANCE_CM = 200     


def get_head_and_feet(landmarks):
    head = landmarks[mp_pose.PoseLandmark.NOSE]
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    foot = type("Point", (object,), {})()  # simple object to hold averaged values
    foot.x = (left_foot.x + right_foot.x) / 2
    foot.y = (left_foot.y + right_foot.y) / 2
    foot.z = (left_foot.z + right_foot.z) / 2
    return head, foot

def compute_norm_height(head, foot):
    head_arr = np.array([head.x, head.y, head.z])
    foot_arr = np.array([foot.x, foot.y, foot.z])
    return np.linalg.norm(foot_arr - head_arr)

def compute_pixel_height(head, foot, image_width, image_height):
    head_px = np.array([head.x * image_width, head.y * image_height])
    foot_px = np.array([foot.x * image_width, foot.y * image_height])
    return np.linalg.norm(foot_px - head_px)

def detect_and_crop_face(image):
    detector = MTCNN()
    results = detector.detect_faces(image)
    if results:
        # Get first detected face
        x, y, w, h = results[0]['box']
        margin = 0.2  # 20% margin
        x1 = max(int(x - margin * w), 0)
        y1 = max(int(y - margin * h), 0)
        x2 = min(int(x + w + margin * w), image.shape[1])
        y2 = min(int(y + h + margin * h), image.shape[0])
        return image[y1:y2, x1:x2]
    else:
        return image

def calibrate_reference(reference_image_path):
    ref_image = cv2.imread(reference_image_path)
    if ref_image is None:
        raise IOError("Error: Could not load reference image.")
    ref_h, ref_w = ref_image.shape[:2]
    ref_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    results = pose.process(ref_rgb)
    if not results.pose_landmarks:
        raise ValueError("No pose detected in reference image.")
    
    landmarks = results.pose_landmarks.landmark
    head, foot = get_head_and_feet(landmarks)
    norm_height = compute_norm_height(head, foot)
    pixel_height = compute_pixel_height(head, foot, ref_w, ref_h)
    
    focal_length = (pixel_height * REFERENCE_DISTANCE_CM) / REFERENCE_HEIGHT
    scaling_factor_ref = REFERENCE_HEIGHT / norm_height
    
    print(f"[Calibration] Reference pixel height: {pixel_height:.2f} pixels, Focal Length: {focal_length:.2f}")
    return focal_length, scaling_factor_ref, norm_height

# ITO YUNG FUNCTION PARA SA NEW IMAGE
def process_new_image(new_image_path, focal_length, new_distance_cm):
    # Nitetest ko lang kung may image ba
    img = cv2.imread(new_image_path)
    if img is None:
        raise Exception("Image not found. Please check the path: " + new_image_path)
    # Kinukuha ko yung height and width ng image
    new_h, new_w = img.shape[:2]
    
    # Cri-nrops ko yung face sa image
    face_img = detect_and_crop_face(img)
    face_img = cv2.resize(face_img, (224, 224))
    # Kinukuha ko yung RGB ng face image
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # ITO YUNG PAGDEDETECT NG GENDER AND RACE
    objs = DeepFace.analyze(
        img_path=face_img_rgb,
        actions=['gender', 'race'],
        enforce_detection=False,
        detector_backend='mtcnn'
    )
    # KINUKUHA KO YUNG DOMINANT OR  YUNG HIGHEST CONFIDENCE
    predicted_gender = max(objs[0]['gender'], key=objs[0]['gender'].get)
    predicted_race = objs[0]['dominant_race']
    
    # ITO YUNG PAGPROCESS NG POSE (OR YUNG MGA LANDMARK SWA KATAWAN) HINDI ITO NAKA-RESIZED
    new_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(new_rgb)
    if not results.pose_landmarks:
        raise ValueError("No pose detected in new image.")
    
    landmarks = results.pose_landmarks.landmark
    # PAG COMPUTE NG HEAD AND FOOT
    head, foot = get_head_and_feet(landmarks)
    # PAG COMPUTE NG NORMALIZED HEIGHT
    norm_height_new = compute_norm_height(head, foot)
    pixel_height_new = compute_pixel_height(head, foot, new_w, new_h)
    real_height_new = (pixel_height_new * new_distance_cm) / focal_length
    # PAG COMPUTE NG SCALING FACTOR, DITO NAKA-ADJUST YUNG HEIGHT
    scaling_factor_new = real_height_new / norm_height_new
    
    print(f"[New Image] {new_image_path} -- Pixel height: {pixel_height_new:.2f} pixels, Real Height: {real_height_new:.2f} cm")
    return img, landmarks, scaling_factor_new, predicted_gender, predicted_race, real_height_new

# ITO YUNG FUNCTION PARA SA DISTANCE NG DALAWANG POINTS
def dist_3d(a, b):
    return np.linalg.norm(np.array([a.x, a.y, a.z]) - np.array([b.x, b.y, b.z]))

# ITO YUNG FUNCTION PARA SA CIRCUMFERENCE
def circumference_approx(a, b, scaling_factor, factor=1.0):
    width = dist_3d(a, b) * scaling_factor
    return np.pi * width * factor

def compute_measurements(landmarks, scaling_factor):
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    # l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    l_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    r_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    # l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW] 
    r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    l_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
    l_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    
    foot_center = np.array([
        (l_foot.x + r_foot.x) / 2,
        (l_foot.y + r_foot.y) / 2,
        (l_foot.z + r_foot.z) / 2
    ])
    head_position = np.array([nose.x, nose.y, nose.z])
    model_height = np.linalg.norm(foot_center - head_position) * scaling_factor
    
    # ITO YUNG MGA FACTOR PARA SA CIRCUMFERENCE, INIIBA ITO DEPENDE SA PART NG BODY
    waist_factor = 1.8  
    chest_factor = 1.3  
    thigh_factor = 0.8  
    wrist_factor = 0.25  
    neck_factor = 0.45
    hip_factor = 1.2
    
    waist_circ = circumference_approx(l_hip, r_hip, scaling_factor, factor=waist_factor)
    belly_circ = waist_circ * 1.1 
    hip_circ = circumference_approx(l_hip, r_hip, scaling_factor, factor=hip_factor)
    chest_circ = circumference_approx(l_shoulder, r_shoulder, scaling_factor, factor=chest_factor)
    thigh_circ = circumference_approx(l_hip, l_knee, scaling_factor, factor=thigh_factor)
    wrist_circ = circumference_approx(r_wrist, r_elbow, scaling_factor, factor=wrist_factor)
    neck_circ = circumference_approx(l_shoulder, r_shoulder, scaling_factor, factor=neck_factor)
    
    upper_arm_length = dist_3d(l_shoulder, l_wrist) * scaling_factor
    hand_length = dist_3d(l_wrist, l_index) * scaling_factor
    total_arm_length = upper_arm_length + hand_length
    
    foot_length = dist_3d(l_heel, l_foot) * scaling_factor
    adjusted_ankle_length = foot_length
    
    measurements = {
        "Estimated Height (cm)": model_height,
        "Waist Circumference (cm)": waist_circ,
        "Hip Circumference (cm)": hip_circ,
        "Belly Circumference (cm)": belly_circ,
        "Chest Circumference (cm)": chest_circ,
        "Thigh Circumference (cm)": thigh_circ,
        "Wrist Circumference (cm)": wrist_circ,
        "Neck Circumference (cm)": neck_circ,
        "Arm Length (Left) (cm)": total_arm_length,
        "Ankle/Foot Length (Left) (cm)": adjusted_ankle_length
    }
    return measurements

def main():
    reference_image_path = "C:/FaceRace/dataset/146cm.jpg"
    new_image_path = "C:/FaceRace/dataset/167cm.jpg"

    focal_length, _, ref_norm_height = calibrate_reference(reference_image_path)
    new_distance_cm = 200 # Ito yung inaadjust para sa height **Bali kaylangan same distance (layo ng tao sa camera) sa ref image saka new image for more accuracy  
    
    img, landmarks, scaling_factor_new, predicted_gender, predicted_race, real_height_new = process_new_image(new_image_path, focal_length, new_distance_cm)
    
    measurements = compute_measurements(landmarks, scaling_factor_new)
    
    print("Predicted Gender: ", predicted_gender.lower())
    try:
        if predicted_gender.lower() == "man":
            bf_percent = 86.010 * math.log10(measurements["Waist Circumference (cm)"] - measurements["Neck Circumference (cm)"]) - 70.041 * math.log10(measurements["Estimated Height (cm)"]) + 36.76
        else:
            bf_percent = 163.205 * math.log10(measurements["Waist Circumference (cm)"] + measurements["Hip Circumference (cm)"] - measurements["Neck Circumference (cm)"]) - 97.684 * math.log10(measurements["Estimated Height (cm)"]) - 78.387
    except ValueError:
        bf_percent = None

    print(f"Dominant Race: {predicted_race}")
    print(f"Estimated height: {real_height_new:.2f} cm")
    for key, value in measurements.items():
        print(f"{key}: {value:.2f}")
    print(f"Body Fat Percentage: {bf_percent:.2f}%")
    
    annotated_image = img.copy()

    # Yung lines na to yung nag aannotate ng image
    mp_drawing.draw_landmarks(annotated_image, 
                              pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).pose_landmarks, 
                              mp_pose.POSE_CONNECTIONS)
    
    # ITO YUNG TEXT SA IMAGE
    cv2.putText(annotated_image, f"Height: {measurements['Estimated Height (cm)']:.2f} cm", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_image, f"Body Fat: {bf_percent:.1f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # DITO NILILIPAT YUNG ANNOTATED IMAGE SA ISANG WINDOW, BALI NIRESIZED KO LANG PARA MAKITA YUNG BUONG IMAGE
    resized_image = cv2.resize(annotated_image, (500, 500))
    cv2.imshow("N-",resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
