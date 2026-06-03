import pytest

from dag.markdown import MarkdownDocument


def test_no_front_matter():
    doc = MarkdownDocument.parse("Just a {{template}} body")
    assert doc.front_matter == {}
    assert doc.body == "Just a {{template}} body"


def test_basic_front_matter():
    doc = MarkdownDocument.parse(
        '---\nname: custom\nmodel: claude-x\nsystem: "Be terse."\n---\nBody {{x}}\n'
    )
    assert doc.front_matter == {"name": "custom", "model": "claude-x", "system": "Be terse."}
    assert doc.body == "Body {{x}}\n"


def test_multiline_block_scalar():
    # The whole point of using real YAML: multi-line system prompts.
    text = "---\nsystem: |\n  Line one.\n  Line two.\n---\nBody"
    doc = MarkdownDocument.parse(text)
    assert doc.front_matter["system"] == "Line one.\nLine two.\n"
    assert doc.body == "Body"


def test_yaml_typed_and_list_values():
    doc = MarkdownDocument.parse("---\ntemperature: 0.7\ntags: [a, b]\n---\nx")
    assert doc.front_matter == {"temperature": 0.7, "tags": ["a", "b"]}


def test_unterminated_is_all_body():
    text = "---\nname: oops\nno closing delimiter {{x}}"
    doc = MarkdownDocument.parse(text)
    assert doc.front_matter == {}
    assert doc.body == text


def test_empty_front_matter_block():
    doc = MarkdownDocument.parse("---\n---\nbody")
    assert doc.front_matter == {}
    assert doc.body == "body"


def test_non_mapping_front_matter_raises():
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        MarkdownDocument.parse("---\n- just\n- a\n- list\n---\nbody")


def test_from_file(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("---\ntitle: Hi\n---\nHello")
    doc = MarkdownDocument.from_file(p)
    assert doc.front_matter == {"title": "Hi"}
    assert doc.body == "Hello"
