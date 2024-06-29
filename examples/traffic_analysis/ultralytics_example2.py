import argparse
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

class GlobalDetectionsManager:
    def __init__(self) -> None:
        self.global_counts: int = 0

    def update(self, detections: sv.Detections) -> None:
        self.global_counts = len(detections)

class VideoProcessor:
    def __init__(self, source_weights_path: str, source_video_path: str, target_video_path: str = None, confidence_threshold: float = 0.3, iou_threshold: float = 0.7) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        # self.model = YOLO(source_weights_path)
        self.model = YOLO('yolov8l.pt')
        self.tracker = sv.ByteTrack()
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = GlobalDetectionsManager()

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)
        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                count = 0
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
                    count += 1
                    # if count == 25:
                        # break
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        self.detections_manager.update(detections)
        return self.annotate_frame(frame, detections)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = self.bounding_box_annotator.annotate(frame.copy(), detections)
        text_anchor = sv.Point(x=50, y=50)

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.bounding_box_annotator.annotate(
            annotated_frame, detections
        )
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )

        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Total Count: {self.detections_manager.global_counts}",
            text_anchor=text_anchor,
            background_color=COLORS.colors[0]
        )
        return annotated_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Object Tracking and Counting with YOLO and ByteTrack")
    parser.add_argument("--source_weights_path", required=True, help="Path to the source weights file", type=str)
    parser.add_argument("--source_video_path", required=True, help="Path to the source video file", type=str)
    parser.add_argument("--target_video_path", default=None, help="Path to the target video file (output)", type=str)
    parser.add_argument("--confidence_threshold", default=0.3, help="Confidence threshold for the model", type=float)
    parser.add_argument("--iou_threshold", default=0.7, help="IOU threshold for the model", type=float)
    args = parser.parse_args()

    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()
