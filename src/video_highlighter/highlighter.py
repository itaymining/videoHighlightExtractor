#!/usr/bin/env python3
"""
AI Video Highlight Generator

Analyzes videos in one or more directories to:
1. Extract individual highlights from each video
2. Save them separately in a highlights folder
3. Create a combined highlight reel
"""
import os
import json
import logging
from typing import List, Tuple
from pathlib import Path
import random
import shutil
import pickle
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from datetime import datetime
import cv2
import face_recognition

from .face_recognizer import FaceRecognizer

logger = logging.getLogger(__name__)


class VideoHighlightGenerator:
    def __init__(self,
                 input_dirs: List[str],
                 output_dir: str,
                 target_duration: int = 180,
                 min_clip_duration: int = 3,
                 max_clip_duration: int = 10,
                 face_detect_min_duration: int = 1,  # Minimum duration for face detection
                 save_video_highlight_txt_metadata: bool = False,  # Save metadata in txt file
                 has_face_only: bool = False,  # Only include segments with faces
                 highlights_per_video: int = 5,  # Number of highlights to extract per video
                 face_images_dir: str = None,  # Directory containing images of known faces
                 use_face_recongniion: bool = False,  # Use face recognition for analysis
                 keep_sort_processing_order: bool = True,  # Keep the order of processing
                 processing_order: str = "asc",  # Order of processing (asc or desc)
                 pickle_path: str = "known_faces",  # Path to save known faces encodings
                 max_video_duration_to_process: int = 600  # Maximum video duration to process
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
        self.use_face_recongniion = use_face_recongniion
        self.face_images_dir = Path(face_images_dir) if self.use_face_recongniion and use_face_recongniion else None
        self.save_video_highlight_txt_metadata = save_video_highlight_txt_metadata
        self.keep_sort_processing_order = keep_sort_processing_order
        self.processing_order = processing_order
        self.fps = None  # Initialize fps
        self.index = 0  # Initialize index
        self.pickle_path = Path(pickle_path)
        self.metadata_path = self.pickle_path.with_suffix('.meta.json')
        self.max_video_duration_to_process = max_video_duration_to_process
        self.known_faces_encodings = self.load_known_faces() if self.face_images_dir else []
        self.face_recognizer = FaceRecognizer(threshold=0.5, min_matches=3)
        self.has_face_only = self.face_images_dir and self.use_face_recongniion and has_face_only and bool(self.known_faces_encodings)
        # Add your known face encodings
        if self.known_faces_encodings:
            self.face_recognizer.add_known_face(self.known_faces_encodings)
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
            clip = video.with_start(start).with_duration(end)

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
                remove_temp=True,
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
        if self.processing_order == "asc":
            video_files.sort()
        elif self.processing_order == "desc":
            video_files.sort(reverse=True)

        return video_files

    def get_video_clips_from_output_dir(self) -> List[Path]:
        """
        Find all video files in the output directory.

        Returns:
            List[Path]: List of paths to video files
        """
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(self.output_dir.rglob(f"*{ext}"))
        if self.processing_order == "asc":
            video_files.sort()
        elif self.processing_order == "desc":
            video_files.sort(reverse=True)

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
        logger.info(f"Detecting scenes in {video_path.name}...")
        video = cv2.VideoCapture(str(video_path))
        self.fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if video duration exceeds the maximum allowed
        video_duration = total_frames / self.fps

        # Return empty if the video duration exceeds the max_video_duration_to_process
        if video_duration > self.max_video_duration_to_process:
            video.release()
            logger.warning(f"Skipping video {video_path.name} as it exceeds the maximum duration allowed {self.max_video_duration_to_process:.2f}s for processing. {video_duration:.2f}s")
            return []

        scenes = []
        prev_frame = None
        current_scene_start = 0
        frame_count = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if prev_frame is not None:
                # Only appends scenes when a change is detected
                if self.is_scene_change(prev_frame, frame):
                    scene_end = frame_count / self.fps
                    if scene_end - current_scene_start >= self.min_clip_duration:
                        scenes.append((current_scene_start, scene_end))
                    current_scene_start = scene_end

            prev_frame = frame
            frame_count += 1

        # Add the final scene if it meets duration requirement
        final_scene_end = frame_count / self.fps
        if final_scene_end - current_scene_start >= self.min_clip_duration:
            scenes.append((current_scene_start, final_scene_end))

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

    def _generate_metadata(self) -> dict:
        """Generate metadata based on file modification times."""
        metadata = {}
        for image_path in self.face_images_dir.glob("*.jpg"):
            metadata[str(image_path)] = image_path.stat().st_mtime
        return metadata

    def _metadata_matches(self, new_metadata: dict) -> bool:
        """Check if current metadata matches the stored metadata."""
        if not self.metadata_path.exists():
            return False
        with open(self.metadata_path, "r") as f:
            stored_metadata = json.load(f)
        return stored_metadata == new_metadata

    def _save_to_pickle(self, encodings: List[np.ndarray], metadata: dict):
        """Save encodings and metadata to pickle and JSON files."""
        with open(self.pickle_path, "wb") as f:
            pickle.dump(encodings, f)
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f)

    def load_known_faces(self) -> List[np.ndarray]:
        """Load known face encodings, building cache if needed."""
        new_metadata = self._generate_metadata()

        # Load from cache if it exists and metadata matches
        if self.pickle_path.exists() and self._metadata_matches(new_metadata):
            logger.info("Loading known faces cache...")
            with open(self.pickle_path, "rb") as f:
                return pickle.load(f)

        logger.info("Building known faces cache...")
        # Otherwise, generate encodings and update cache
        known_encodings = []
        for image_path in self.face_images_dir.glob("*.jpg"):
            img = face_recognition.load_image_file(str(image_path))
            encoding = face_recognition.face_encodings(img)
            if encoding:
                known_encodings.append(encoding[0])

        # Save to pickle with metadata
        self._save_to_pickle(known_encodings, new_metadata)
        return known_encodings

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
        segment = video.with_start(start).with_duration(end)
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

        if not self.known_faces_encodings or not self.face_images_dir:
            return 0.7 * avg_motion + 0.3 * avg_content, False

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
        if self.known_faces_encodings:
            for frame in frames:
                is_match, confidence = self.face_recognizer.is_person_in_frame(frame)
                if is_match:
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
            self.index += 1
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
                scored_segments.append((start, end, score, self.index))
            elif not self.has_face_only:  # Add all segments
                scored_segments.append((start, end, score, self.index))

        video.close()
        return sorted(scored_segments, key=lambda x: x[2], reverse=True)

    def create_highlight_reel(self):
        """
        Create individual highlights and a combined highlight reel.
        """
        try:
            # File to save the scored segments
            scored_segments_file = self.output_dir / "scored_segments.pkl"
            if os.path.exists(scored_segments_file):
                # Load scored segments from file
                logger.info("Loading scored segments from file...")
                with open(scored_segments_file, 'rb') as f:
                    all_segments = pickle.load(f)
            else:
                video_files = self.find_video_files()
                if not video_files:
                    raise ValueError("No video files found in the specified input directories")

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
                        for highlight_num, (start, end, score, index) in enumerate(
                                scored_segments[:self.highlights_per_video], 1):
                            try:
                                self.save_individual_highlight(
                                    video_path, start, end, highlight_num, score)
                            except Exception as e:
                                logger.error(f"Failed to save highlight {highlight_num} for {video_path}: {e}")
                                continue  # Skip to the next highlight if saving fails

                        # Add to collection for combined highlight reel
                        all_segments.extend(
                            (video_path, start, end, index, score)
                            for start, end, score, index in scored_segments)

                    except Exception as e:
                        logger.error(f"Failed to process video {video_path}: {e}")
                        continue  # Skip to the next video if processing fails
                # Save the scored segments to a file
                logger.info("Saving scored segments to file...")
                with open(scored_segments_file, 'wb') as f:
                    pickle.dump(all_segments, f)

            # Create combined highlight reel
            logger.info("Creating combined highlight reel...")

            # Sort all segments by score
            all_segments.sort(key=lambda x: x[4], reverse=True)
            self.select_and_concatenate_clips(all_segments)

        except Exception as e:
            logger.error(f"Failed to create highlight reel: {str(e)}")
            raise

    def create_highlight_reel_by_existing_clips(self):
        """
        Build a combined highlight reel from clips already in the output directory.
        """
        try:
            # File to save the scored segments
            scored_segments_file = self.output_dir / "scored_segments.pkl"

            if os.path.exists(scored_segments_file):
                # Load scored segments from file
                logger.info("Loading scored segments from file...")
                with open(scored_segments_file, 'rb') as f:
                    all_segments = pickle.load(f)
            else:
                video_files = self.get_video_clips_from_output_dir()
                if not video_files:
                    raise ValueError("No video files found in the specified input directories")

                all_segments = []
                # Analyze all videos and save individual highlights
                logger.info("Analyzing videos and extracting highlights...")
                for video_path in tqdm(video_files):
                    try:
                        # Detect scenes
                        scenes = self.detect_scenes(video_path)

                        # Score and select best segments
                        scored_segments = self.select_best_segments(video_path, scenes)

                        # Add to collection for combined highlight reel
                        all_segments.extend(
                            (video_path, start, end, index, score)
                            for start, end, score, index in scored_segments)

                    except Exception as e:
                        logger.error(f"Failed to process video {video_path}: {e}")
                        continue  # Skip to the next video if processing fails
                # Save the scored segments to a file
                logger.info("Saving scored segments to file...")
                with open(scored_segments_file, 'wb') as f:
                    pickle.dump(all_segments, f)

            # Create combined highlight reel
            logger.info("Creating combined highlight reel...")

            # Sort all segments by score
            all_segments.sort(key=lambda x: x[4], reverse=True)
            self.select_and_concatenate_clips(all_segments)

        except Exception as e:
            logger.error(f"Failed to create highlight reel: {str(e)}")
            raise

    def select_and_concatenate_clips(self, all_segments):
        """
        Select top segments until the target duration is reached and concatenate them.
        """
        selected_clips = []
        total_duration = 0
        for video_path, start, end, index, score in all_segments:
            clip_duration = end - start
            if total_duration + clip_duration > self.target_duration:
                break
            try:
                clip = VideoFileClip(str(video_path)).with_start(start).with_duration(end)
                selected_clips.append((clip, start, end, index, score, video_path))
                total_duration += clip_duration
            except Exception as e:
                logger.error(f"Failed to add clip from {video_path} to combined reel: {e}")
                continue  # Skip to the next segment if adding fails

        if self.keep_sort_processing_order:
            if self.processing_order == "asc":
                # Sort selected clips by their original indices
                selected_clips.sort(key=lambda x: x[3])
            elif self.processing_order == "desc":
                # Sort selected clips by their original indices in reverse
                selected_clips.sort(key=lambda x: x[3], reverse=True)
        else:
            # Randomly shuffle the selected clips
            random.shuffle(selected_clips)

        # Extract only the clips for concatenation
        target_width = 1920
        target_height = 1080
        selected_clips = [clip for clip, _, _, _, _, _ in selected_clips]

        if not selected_clips:
            logger.warning("No clips available for processing.")
            return None

        # Get the current date in a specific format, e.g., YYYY-MM-DD
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Create a clips directory with the current date in the name if it doesn't exist
        clips_dir = self.output_dir / f"individual_clips_{current_date}"
        clips_dir.mkdir(parents=True, exist_ok=True)

        resized_clips = []
        for index, clip in enumerate(selected_clips, 1):
            clip_width, clip_height = clip.size

            if clip_width > clip_height:  # Landscape
                resized_clip = clip.resized(clip, width=target_width, height=target_height)
            else:  # Portrait
                resized_clip = clip.resized(clip, height=target_height)
                letterbox_width = target_width - resized_clip.size[0]
                if letterbox_width > 0:
                    resized_clip = resized_clip.margin(
                        left=letterbox_width // 2,
                        right=letterbox_width // 2
                    )

            # Save individual clip
            clip_filename = f"clip_{index:03d}.mp4"  # This will create names like clip_001.mp4
            clip_path = clips_dir / clip_filename
            resized_clip.write_videofile(
                str(clip_path),
                codec='libx264',
                audio_codec='aac',
                bitrate='8000k'
            )
            logger.info(f"Saved individual clip {index} at: {clip_path}")
            logger.info(f"Progress: {index}/{len(selected_clips)}")
            resized_clips.append(resized_clip)

        # Concatenate and save the combined highlight reel
        final_reel = concatenate_videoclips(resized_clips, method="compose")
        output_reel_path = self.output_dir / "combined_highlight_reel.mp4"
        final_reel.write_videofile(
            str(output_reel_path),
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k'
        )
        logger.info(f"Combined highlight reel saved at: {output_reel_path}")

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
