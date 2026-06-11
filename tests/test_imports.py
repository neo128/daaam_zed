def test_package_imports():
    import daaam_zed

    assert daaam_zed.__version__


def test_capture_settings_defaults():
    from daaam_zed.models import CaptureSettings

    settings = CaptureSettings()

    assert settings.resolution == "HD720"
    assert settings.fps == 15
    assert settings.depth_mode == "ULTRA"
    assert settings.output_width == 640
    assert settings.output_height == 480
    assert settings.resize_mode == "crop_resize"
