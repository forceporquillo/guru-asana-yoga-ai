import os.path
from typing import List, Optional, Tuple
from urllib.error import HTTPError

from matplotlib import pyplot as plt
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec, _normalize_color


def show_image(img, figsize=(10, 10)):
    try:
        """Shows output PIL image."""
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()
    except HTTPError:
        pass


_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


def draw_plot_landmarks_save(
        difficulty_level,
        pose_class_name,
        image_name,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        landmark_drawing_spec: DrawingSpec = DrawingSpec(
            color=RED_COLOR, thickness=5),
        connection_drawing_spec: DrawingSpec = DrawingSpec(
            color=BLACK_COLOR, thickness=5),
        elevation: int = 10,
        azimuth: int = 10):
    if not landmark_list:
        return
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        ax.scatter3D(
            xs=[-landmark.z],
            ys=[landmark.x],
            zs=[-landmark.y],
            color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness)
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness)
    # plt.show()

    path = os.path.join('pose_landmark_3d_plot', difficulty_level, pose_class_name)
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, image_name + '.png'), dpi=100)
    plt.close()
