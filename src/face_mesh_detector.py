import cv2
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self, max_faces=1,refine_landmarks=True):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        # Initialize the Face Mesh model
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces = max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self,frame):
        #bgr -> rgb
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        return result
    
    def draw(self,frame,result):
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_styles.get_default_face_mesh_tesselation_style()
                )
        return frame