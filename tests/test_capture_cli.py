from argparse import Namespace
from pathlib import Path


def test_cli_args_map_to_capture_settings():
    from scripts.capture_zed_to_daaam_dataset import build_settings_from_args

    settings = build_settings_from_args(
        Namespace(
            resolution="HD1080",
            fps=30,
            depth_mode="NEURAL",
            coordinate_system="IMAGE",
            svo_file=Path("/tmp/input.svo"),
            width=800,
            height=600,
            resize_mode="resize",
            max_frames=120,
            duration_sec=5.0,
            stride=2,
            warmup_frames=10,
            min_depth_m=0.2,
            max_depth_m=8.0,
            keep_invalid_depth=True,
            drop_bad_tracking=False,
            area_memory=False,
            png_compression=1,
            record_svo=True,
        )
    )

    assert settings.resolution == "HD1080"
    assert settings.fps == 30
    assert settings.depth_mode == "NEURAL"
    assert settings.svo_file == Path("/tmp/input.svo")
    assert settings.output_width == 800
    assert settings.output_height == 600
    assert settings.resize_mode == "resize"
    assert settings.max_frames == 120
    assert settings.duration_sec == 5.0
    assert settings.stride == 2
    assert settings.warmup_frames == 10
    assert settings.min_depth_m == 0.2
    assert settings.max_depth_m == 8.0
    assert settings.keep_invalid_depth is True
    assert settings.drop_bad_tracking is False
    assert settings.area_memory is False
    assert settings.png_compression == 1
    assert settings.enable_svo_recording is True
