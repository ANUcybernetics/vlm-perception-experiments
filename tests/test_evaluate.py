from vlm_perception.evaluate import _parse_response
from vlm_perception.models import Side


def test_parse_json_left():
    assert _parse_response('{"answer": "left"}') == Side.left


def test_parse_json_right():
    assert _parse_response('Sure! {"answer": "right"}') == Side.right


def test_parse_freetext_left():
    assert _parse_response("The left circle is in front.") == Side.left


def test_parse_freetext_right():
    assert _parse_response("The right circle occludes the other.") == Side.right


def test_parse_ambiguous_returns_none():
    assert _parse_response("The left and right circles overlap.") is None


def test_parse_gibberish_returns_none():
    assert _parse_response("I cannot determine the answer.") is None
