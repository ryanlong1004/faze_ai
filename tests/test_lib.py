from src.lib import parse_timestamps


def test_parse_timestamps():
    text_with_timestamps = "Hello <|0.5|> world <|1.0|>!"
    expected_timestamps = [0.5, 1.0]
    expected_text_segments = ["Hello ", " world ", "!"]

    timestamps, text_segments = parse_timestamps(text_with_timestamps)

    assert timestamps == expected_timestamps
    assert text_segments == expected_text_segments


def test_parse_timestamps_empty():
    text_with_timestamps = ""
    expected_timestamps = []
    expected_text_segments = []

    timestamps, text_segments = parse_timestamps(text_with_timestamps)

    assert timestamps == expected_timestamps
    assert text_segments == expected_text_segments


def test_parse_timestamps_no_timestamps():
    text_with_timestamps = "Hello world!"
    expected_timestamps = []
    expected_text_segments = ["Hello world!"]

    timestamps, text_segments = parse_timestamps(text_with_timestamps)

    assert timestamps == expected_timestamps
    assert text_segments == expected_text_segments
