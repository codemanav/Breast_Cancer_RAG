import pytest

from src.utils import parse_json_answer


@pytest.mark.parametrize(
    "text",
    [
        '{"relative_risk": "higher"}',
        '```json\n{"relative_risk": "higher"}\n```',
        'Here is the analysis:\n{"relative_risk": "higher"}\nHope this helps.',
    ],
    ids=["bare", "code-fence", "surrounding-prose"],
)
def test_parses_json_in_common_wrappers(text):
    assert parse_json_answer(text) == {"relative_risk": "higher"}


def test_parses_nested_structures():
    text = '{"risk_factors": [{"factor": "stress", "who": "Sue"}], "relative_risk": ""}'
    assert parse_json_answer(text)["risk_factors"][0]["who"] == "Sue"


@pytest.mark.parametrize("text", ["no json here", "{not: valid json}", ""])
def test_returns_none_when_unparseable(text):
    assert parse_json_answer(text) is None
