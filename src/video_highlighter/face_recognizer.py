"""Known-face recognition used to score video segments."""
from typing import List, Tuple

import numpy as np
import cv2
import face_recognition


class FaceRecognizer:
    def __init__(self, threshold: float = 0.5, min_matches: int = 3):
        self.known_faces_encodings: List[np.ndarray] = []
        self.threshold = threshold
        self.min_matches = min_matches

    def add_known_face(self, face_encodings: List[np.ndarray]) -> None:
        """Add known face encodings to the recognizer."""
        self.known_faces_encodings.extend(face_encodings)

    def is_person_in_frame(self, frame: np.ndarray, min_face_size: int = 30) -> Tuple[bool, float]:
        """
        Check if the known person appears in the frame.
        Returns: Tuple of (is_match: bool, confidence: float)
        """
        # Convert to RGB (face_recognition expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")

        # Filter out small faces that are likely too far/small to be reliable
        face_locations = [loc for loc in face_locations
                          if (loc[2] - loc[0]) >= min_face_size and (loc[1] - loc[3]) >= min_face_size]

        if not face_locations:
            return False, 0.0

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            # Calculate distances to all known encodings
            distances = face_recognition.face_distance(self.known_faces_encodings, face_encoding)

            # Count how many known faces match below threshold
            matches = sum(1 for distance in distances if distance < self.threshold)

            # Calculate average distance for matches
            matching_distances = [d for d in distances if d < self.threshold]
            if matching_distances:
                avg_distance = np.mean(matching_distances)
                confidence = 1 - avg_distance  # Convert distance to confidence score

                # Return true if we have enough matches
                if matches >= self.min_matches:
                    return True, confidence

        return False, 0.0

    def find_best_threshold(self, validation_frames: List[np.ndarray],
                            validation_labels: List[bool]) -> float:
        """
        Find the optimal threshold using validation data.
        validation_frames: List of frames to test
        validation_labels: True/False for each frame indicating if person is present
        """
        best_threshold = self.threshold
        best_f1 = 0.0

        for threshold in np.arange(0.3, 0.7, 0.05):
            self.threshold = threshold
            predictions = [self.is_person_in_frame(frame)[0] for frame in validation_frames]

            # Calculate F1 score
            tp = sum(1 for p, l in zip(predictions, validation_labels) if p and l)
            fp = sum(1 for p, l in zip(predictions, validation_labels) if p and not l)
            fn = sum(1 for p, l in zip(predictions, validation_labels) if not p and l)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold
