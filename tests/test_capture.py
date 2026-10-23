from vlmforge.capture import mask, parse_channels


def test_channels_and_mask():
    assert parse_channels("1-3,7") == [1, 2, 3, 7]
    assert mask("rtsp://admin:s3cret@10.0.0.1:554/x") == "rtsp://admin:***@10.0.0.1:554/x"
