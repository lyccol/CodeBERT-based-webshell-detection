from webshell_detection.preprocessing import (
    collapse_whitespace,
    preprocess_php_code,
    remove_php_comments,
    strip_php_tags,
)


def test_strip_php_tags_removes_php_wrappers():
    code = "<?php echo 'test'; ?>"
    assert "<?php" not in strip_php_tags(code)


def test_remove_php_comments_eliminates_single_and_multi_line():
    code = """<?php
// comment
# another
echo 'hi'; /* block */
?>"""
    cleaned = remove_php_comments(code)
    assert "comment" not in cleaned
    assert "another" not in cleaned
    assert "block" not in cleaned


def test_collapse_whitespace_reduces_sequences():
    text = "a\n\n  b\t\t c"
    assert collapse_whitespace(text) == "a b c"


def test_preprocess_php_code_applies_all_steps():
    code = """<?php
// comment
echo $var; /* block */
?>"""
    cleaned = preprocess_php_code(code, max_characters=20)
    assert "<?php" not in cleaned
    assert "comment" not in cleaned
    assert len(cleaned) <= 20
