#!/usr/bin/env python3
"""
AI Video Highlight Generator

This script analyzes videos in a directory to:
1. Extract individual highlights from each video
2. Save them separately in a highlights folder
3. Create a combined highlight reel
"""

import os
import logging
from typing import List, Tuple, Dict
from pathlib import Path
import random
import shutil
import face_recognition
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from scenedetect import detect, ContentDetector
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoHighlightGenerator:
    def __init__(self, 
                 input_dirs: List[str],  # Changed from str to List[str]
                 output_dir: str,
                 target_duration: int = 180,
                 min_clip_duration: int = 3,
                 max_clip_duration: int = 10,
                 face_detect_min_duration: int = 1,  # Minimum duration for face detection
                 save_video_highlight_txt_metadata: bool = False,  # Save metadata in txt file
                 has_face_only: bool = False,  # Only include segments with faces
                 highlights_per_video: int = 5, # Number of highlights to extract per video
                 face_images_dir: str = None  # Directory containing images of known faces
                 ):
        """
        Initialize the highlight generator.
        
        Args:
            input_dirs (List[str]): List of directories containing input videos
            output_dir (str): Directory for output videos
            target_duration (int): Target duration for the combined highlight reel
            min_clip_duration (int): Minimum duration for each clip
            max_clip_duration (int): Maximum duration for each clip
            highlights_per_video (int): Number of top highlights to save per video
        """
        self.input_dirs = [Path(input_dir) for input_dir in input_dirs]
        self.output_dir = Path(output_dir)
        self.highlights_dir = self.output_dir / "individual_highlights"
        self.target_duration = target_duration
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.face_detect_min_duration = face_detect_min_duration
        self.highlights_per_video = highlights_per_video
        self.face_images_dir = Path(face_images_dir) if face_images_dir else None
        self.known_faces_encodings = self.load_known_faces() if self.face_images_dir else []
        self.has_face_only = has_face_only and bool(self.known_faces_encodings)
        self.fps = None  # Initialize fps
        # Initialize AI models
        self.content_analyzer = ResNet50(weights='imagenet', include_top=True)
        
        # Supported video formats
        self.video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        
        # Create output directories
        self.setup_directories()

    def setup_directories(self):
        """Create necessary output directories."""
        self.output_dir.mkdir(exist_ok=True)
        self.highlights_dir.mkdir(exist_ok=True)
        
        # Clear previous highlights
        if any(self.highlights_dir.iterdir()):
            user_input = input("Previous highlights found. Clear them? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.highlights_dir)
                self.highlights_dir.mkdir()

    def get_safe_filename(self, video_path: Path, highlight_num: int) -> str:
        """
        Create a safe filename for the highlight video.
        
        Args:
            video_path (Path): Original video path
            highlight_num (int): Highlight number
            
        Returns:
            str: Safe filename
        """
        # Remove extension and get base name
        base_name = video_path.stem
        # Replace unsafe characters
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
        return f"{safe_name}_highlight_{highlight_num:02d}.mp4"

    def save_individual_highlight(self, 
                                video_path: Path,
                                start: float,
                                end: float,
                                highlight_num: int,
                                score: float):
        """
        Save an individual highlight clip.
        
        Args:
            video_path (Path): Path to original video
            start (float): Start time of highlight
            end (float): End time of highlight
            highlight_num (int): Highlight number for this video
            score (float): Highlight score
        """
        try:
            video = VideoFileClip(str(video_path))
            clip = video.subclip(start, end)
            
            # Create filename
            output_filename = self.get_safe_filename(video_path, highlight_num)
            output_path = self.highlights_dir / output_filename
            
            # Save metadata
            metadata_path = output_path.with_suffix('.txt')

            if self.save_video_highlight_txt_metadata:
                with open(metadata_path, 'w') as f:
                    f.write(f"Original video: {video_path.name}\n")
                    f.write(f"Highlight {highlight_num}\n")
                    f.write(f"Start time: {start:.2f}s\n")
                    f.write(f"End time: {end:.2f}s\n")
                    f.write(f"Duration: {end-start:.2f}s\n")
                    f.write(f"Interest score: {score:.3f}\n")
            
            # Save video
            logger.info(f"Saving highlight: {output_filename}")
            clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            clip.close()
            video.close()
            
        except Exception as e:
            logger.error(f"Failed to save highlight {output_filename}: {str(e)}")
            raise

    def find_video_files(self) -> List[Path]:
        """
        Find all video files in the input directories.
        
        Returns:
            List[Path]: List of paths to video files
        """
        video_files = []
        for input_dir in self.input_dirs:
            for ext in self.video_extensions:
                video_files.extend(input_dir.rglob(f"*{ext}"))
        return video_files
    
    def analyze_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Analyze motion between two frames using optical flow.
        
        Args:
            frame1 (np.ndarray): First frame
            frame2 (np.ndarray): Second frame
            
        Returns:
            float: Motion score between 0 and 1
        """
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate magnitude of flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Return normalized average magnitude
        return np.mean(magnitude) / 100.0  # Normalized to roughly 0-1 range
    
    def analyze_content(self, frame: np.ndarray) -> float:
        """
        Analyze frame content using ResNet50 for interestingness.
        
        Args:
            frame (np.ndarray): Frame to analyze
            
        Returns:
            float: Content interest score between 0 and 1
        """
        # Resize and preprocess frame
        img = cv2.resize(frame, (224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # Get predictions
        predictions = self.content_analyzer.predict(img, verbose=0)
        
        # Calculate "interestingness" based on prediction confidence
        # Higher confidence in top predictions indicates more interesting content
        top_5_confidence = np.sort(predictions[0])[-5:]
        return float(np.mean(top_5_confidence))
    
    def detect_scenes(self, video_path: Path) -> List[Tuple[float, float]]:
        """
        Detect scene changes in video.
        
        Args:
            video_path (Path): Path to video file
            
        Returns:
            List[Tuple[float, float]]: List of (start_time, end_time) tuples
        """
        video = cv2.VideoCapture(str(video_path))
        self.fps = video.get(cv2.CAP_PROP_FPS)
        
        scenes = []
        prev_frame = None
        current_scene_start = 0
        frame_count = 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            if prev_frame is not None:
                # Check for scene change using content detector
                if self.is_scene_change(prev_frame, frame):
                    scene_end = frame_count / self.fps
                    if scene_end - current_scene_start >= self.min_clip_duration:
                        scenes.append((current_scene_start, scene_end))
                    current_scene_start = scene_end
            
            prev_frame = frame
            frame_count += 1
        
        video.release()
        return scenes
    
    def is_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """
        Detect if there is a scene change between two frames.
        
        Args:
            frame1 (np.ndarray): First frame
            frame2 (np.ndarray): Second frame
            
        Returns:
            bool: True if scene change detected
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(gray1, gray2)
        mean_diff = np.mean(diff)
        
        # Threshold for scene change
        return mean_diff > 30
    
    def load_known_faces(self) -> List[np.ndarray]:
        """Load known face encodings from face_images_dir."""
        known_encodings = []
        for image_path in self.face_images_dir.glob("*.jpg"):
            img = face_recognition.load_image_file(str(image_path))
            encoding = face_recognition.face_encodings(img)
            if encoding:
                known_encodings.append(encoding[0])
        return known_encodings

    def is_person_in_frame(self, frame: np.ndarray) -> bool:
        """Check if the known person appears in the frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding)
            if any(matches):
                return True
        return False

    def analyze_segment(self, video: VideoFileClip, start: float, end: float) -> float:
        """
        Analyze a video segment for interesting content.
        
        Args:
            video (VideoFileClip): Video clip
            start (float): Start time in seconds
            end (float): End time in seconds
            
        Returns:
            float: Segment interest score between 0 and 1
        """
        segment = video.subclip(start, end)
        frames = list(segment.iter_frames())
        
        if len(frames) < 2:
            return 0.0
        
        # Analyze motion
        motion_scores = []
        for i in range(len(frames) - 1):
            motion_score = self.analyze_motion(frames[i], frames[i + 1])
            motion_scores.append(motion_score)
        
        # Analyze content
        content_scores = []
        sample_frames = frames[::max(1, len(frames) // 5)]  # Sample every nth frame
        for frame in sample_frames:
            content_score = self.analyze_content(frame)
            content_scores.append(content_score)
        
        # Combine scores (weight motion more heavily for highlights)
        avg_motion = np.mean(motion_scores) if motion_scores else 0
        avg_content = np.mean(content_scores) if content_scores else 0
        
        # Face presence analysis
        face_duration = self.analyze_face_presence(frames)  # method to analyze face presence
        person_score = self.calculate_person_score(face_duration)  # Calculate score based on face presence
        has_face = face_duration > self.face_detect_min_duration  # Consider it a face if detected for more than 1 second
        return 0.5 * avg_motion + 0.2 * avg_content + 0.3 * person_score, has_face

    def analyze_face_presence(self, frames):
        """
        Analyze the presence of known faces in the frames and return the duration they appear.
        
        Args:
            frames (list): List of frames to analyze
            
        Returns:
            float: Duration in seconds that known faces are present in the frames
        """
        duration = 0
        for frame in frames:
            if self.known_faces_encodings and self.is_person_in_frame(frame):
                duration += 1 / self.fps  # Assuming each frame represents 1/fps seconds
                
        return duration

    def calculate_person_score(self, face_duration):
        """
        Calculate the person score based on the duration of face presence.
        
        Args:
            face_duration (float): Duration in seconds
            
        Returns:
            float: Score based on face presence duration
        """
        # Normalize face duration to a score between 0 and 1 (assuming self.max_clip_duration is the longest you'd expect)
        normalized_duration = min(face_duration / self.max_clip_duration, self.min_clip_duration)
        
        # Add bonus if a known face is detected in the frame
        bonus = 0.2 if face_duration > 0 else 0.0  # Bonus if any known face is detected
        score = normalized_duration + bonus  # Base score based on duration plus bonus
        
        return score

    def select_best_segments(self, 
                           video_path: Path,
                           scenes: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        """
        Select the best segments from the video based on analysis.
        
        Args:
            video_path (Path): Path to video file
            scenes (List[Tuple[float, float]]): List of scene timestamps
            
        Returns:
            List[Tuple[float, float, float]]: List of (start_time, end_time, score)
        """
        video = VideoFileClip(str(video_path))
        scored_segments = []
        
        for start, end in scenes:
            # Limit segment duration
            duration = end - start
            if duration > self.max_clip_duration:
                end = start + self.max_clip_duration
            elif duration < self.min_clip_duration:
                continue
            
            # Analyze segment
            score, has_face = self.analyze_segment(video, start, end)
            # Store segments that have faces
            if self.has_face_only and has_face:  # Only add segments that contain a face
                scored_segments.append((start, end, score))
            elif not self.has_face_only:  # Add all segments
                scored_segments.append((start, end, score))
        
        video.close()
        return sorted(scored_segments, key=lambda x: x[2], reverse=True)

    def create_highlight_reel(self):
        """
        Create individual highlights and a combined highlight reel.
        """
        try:
            video_files = self.find_video_files()
            if not video_files:
                raise ValueError(f"No video files found in the specified input directories")

            all_segments = []

            # Analyze all videos and save individual highlights
            logger.info("Analyzing videos and extracting highlights...")
            for video_path in tqdm(video_files):
                try:
                    # Detect scenes
                    scenes = self.detect_scenes(video_path)
                    
                    # Score and select best segments
                    scored_segments = self.select_best_segments(video_path, scenes)
                    
                    # Save top highlights for this video
                    for highlight_num, (start, end, score) in enumerate(
                        scored_segments[:self.highlights_per_video], 1):
                        try:
                            self.save_individual_highlight(
                                video_path, start, end, highlight_num, score)
                        except Exception as e:
                            logger.error(f"Failed to save highlight {highlight_num} for {video_path}: {e}")
                            continue  # Skip to the next highlight if saving fails

                    # Add to collection for combined highlight reel
                    all_segments.extend(
                        (video_path, start, end, score) 
                        for start, end, score in scored_segments)

                except Exception as e:
                    logger.error(f"Failed to process video {video_path}: {e}")
                    continue  # Skip to the next video if processing fails

            # Create combined highlight reel
            logger.info("Creating combined highlight reel...")
            
            # Sort all segments by score
            all_segments.sort(key=lambda x: x[3], reverse=True)
            
            # Select top segments until the target duration is reached
            selected_clips = []
            total_duration = 0
            for video_path, start, end, score in all_segments:
                clip_duration = end - start
                if total_duration + clip_duration > self.target_duration:
                    break
                try:
                    clip = VideoFileClip(str(video_path)).subclip(start, end)
                    selected_clips.append(clip)
                    total_duration += clip_duration
                except Exception as e:
                    logger.error(f"Failed to add clip from {video_path} to combined reel: {e}")
                    continue  # Skip to the next segment if adding fails

            # Concatenate and save the combined highlight reel
            if selected_clips:
                final_reel = concatenate_videoclips(selected_clips, method="compose")
                output_reel_path = self.output_dir / "combined_highlight_reel.mp4"
                final_reel.write_videofile(
                    str(output_reel_path), codec='libx264', audio_codec='aac'
                )
                logger.info(f"Combined highlight reel saved at: {output_reel_path}")
            else:
                logger.warning("No clips available for the combined highlight reel.")

        except Exception as e:
            logger.error(f"An error occurred while creating the highlight reel: {e}")


    def create_summary_report(self):
        """Create a summary report of all extracted highlights."""
        report_path = self.output_dir / "highlights_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("Video Highlights Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Group highlights by original video
            highlights = {}
            for path in self.highlights_dir.glob("*.mp4"):
                video_name = path.stem.split("_highlight_")[0]
                if video_name not in highlights:
                    highlights[video_name] = []
                highlights[video_name].append(path)
            
            # Write summary for each video
            for video_name, highlight_paths in highlights.items():
                f.write(f"\nOriginal Video: {video_name}\n")
                f.write("-" * 30 + "\n")
                
                for highlight_path in sorted(highlight_paths):
                    # Get metadata from corresponding txt file
                    metadata_path = highlight_path.with_suffix('.txt')
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as mf:
                            metadata = mf.read()
                        f.write(f"\n{highlight_path.name}:\n{metadata}\n")
                
                f.write("\n" + "=" * 50 + "\n")

def main():
    """
    Main function to run the highlight generator.
    """
    # Configuration
    INPUT_DIRS = ["2024-10"]  # List of input directories
    OUTPUT_DIR = "video_highlights5"
    TARGET_DURATION = 180  # 3 minutes
    MIN_CLIP_DURATION = 3  # Minimum duration for each clip
    MAX_CLIP_DURATION = 10  # Maximum duration for each clip
    FACE_DETECT_MIN_DURATION = 1  # Minimum duration for face detection
    HAS_FACE_ONLY = True  # Only include segments with faces
    HIGHLIGHTS_PER_VIDEO = 5  # Number of highlights to extract per video
    FACE_RECOGNITION_DIR = "faces"  # Directory containing images of known faces
    SAVE_VIDEO_HIGHLIGHT_TXT_METADATA = False  # Save metadata in txt file
    
    try:
        # Create highlight generator
        generator = VideoHighlightGenerator(
            input_dirs=INPUT_DIRS,
            output_dir=OUTPUT_DIR,
            target_duration=TARGET_DURATION,
            min_clip_duration = MIN_CLIP_DURATION,
            max_clip_duration = MAX_CLIP_DURATION,
            face_detect_min_duration = FACE_DETECT_MIN_DURATION,
            save_video_highlight_txt_metadata = SAVE_VIDEO_HIGHLIGHT_TXT_METADATA,
            has_face_only = HAS_FACE_ONLY,
            highlights_per_video= HIGHLIGHTS_PER_VIDEO, 
            face_images_dir=FACE_RECOGNITION_DIR
        )
        
        # Generate highlights
        generator.create_highlight_reel()
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

    #pip install moviepy opencv-python scenedetect tensorflow numpy pillow tqdm dlib
