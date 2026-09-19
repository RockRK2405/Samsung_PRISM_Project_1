"""Video → sampled frames → cropped-face pipeline.

``FrameExtractor`` is dependency-light (OpenCV only) and used by BOTH the
face track and the general AI-video track, so it always imports. The
face-specific pieces need facenet-pytorch / mediapipe, which the video
track does not require — they are guarded so the module imports cleanly
on machines (e.g. a lean Colab/Kaggle GPU box) without the face deps.
"""

from src.preprocessing.frame_extractor import FrameExtractor

__all__ = ["FrameExtractor"]

try:
    from src.preprocessing.celeb_df import (
        CelebDFRecord,
        parse_test_list,
        scan_celeb_df_training,
    )
    from src.preprocessing.face_detector import MTCNNFaceDetector, save_crop
    from src.preprocessing.manifest import (
        FF_MANIPULATIONS,
        FF_REAL_FOLDER,
        VideoRecord,
        group_split,
        scan_ff_c23,
        write_manifest,
    )

    __all__ += [
        "CelebDFRecord",
        "FF_MANIPULATIONS",
        "FF_REAL_FOLDER",
        "MTCNNFaceDetector",
        "VideoRecord",
        "group_split",
        "parse_test_list",
        "save_crop",
        "scan_celeb_df_training",
        "scan_ff_c23",
        "write_manifest",
    ]
except ImportError:
    # Face-track deps (facenet-pytorch / mediapipe) not installed — fine for
    # the general AI-video track.
    pass
