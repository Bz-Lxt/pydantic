"""Tests for mapping validation errors back to positions in the original JSON input.

Positions are requested with the (off by default) ``include_position=True`` argument to
``ValidationError.errors`` / ``ValidationError.json``.
"""

import copy
import json
import pickle

import pytest

from pydantic_core import SchemaValidator
from pydantic_core import core_schema as s


def positions(validator, data, **kwargs):
    with pytest.raises(Exception) as exc_info:
        validator.validate_json(data, **kwargs)
    return exc_info.value.errors(include_position=True, include_url=False)


def line_col(data, byte_offset):
    """Reference implementation of the documented line/column semantics (1-based, code points)."""
    prefix = data.encode()[:byte_offset].decode()
    line = prefix.count('\n') + 1
    column = len(prefix.rsplit('\n', 1)[-1]) + 1
    return line, column


def check_position(data, position, expected_fragment):
    # byte offsets are a half-open [start, end) range over the exact JSON bytes
    raw = data.encode()
    assert raw[position['start'] : position['end']].decode() == expected_fragment
    # line/column agree with the byte offsets (columns count Unicode code points)
    assert (position['start_line'], position['start_column']) == line_col(data, position['start'])
    assert (position['end_line'], position['end_column']) == line_col(data, position['end'])


def simple_validator():
    return SchemaValidator(
        s.typed_dict_schema(
            {
                'name': s.typed_dict_field(s.str_schema()),
                'age': s.typed_dict_field(s.int_schema()),
                'items': s.typed_dict_field(s.list_schema(s.int_schema()), required=False),
            },
            extra_behavior='forbid',
        )
    )


def test_position_is_opt_in():
    validator = simple_validator()
    with pytest.raises(Exception) as exc_info:
        validator.validate_json('{"age": "x"}')
    assert 'position' not in exc_info.value.errors()[0]
    assert 'position' not in exc_info.value.errors(include_position=False)[0]
    position = exc_info.value.errors(include_position=True)[0]['position']
    assert set(position) == {'start', 'end', 'start_line', 'start_column', 'end_line', 'end_column'}


def test_nested_scalar_and_fragment_slice():
    validator = simple_validator()
    data = '{\n  "name": "Bob",\n  "age":   "x",\n  "items": [1, 2, "bad"]\n}'
    errors = positions(validator, data)
    assert [e['loc'] for e in errors] == [('age',), ('items', 2)]

    age, item = errors
    check_position(data, age['position'], '"x"')
    assert age['position']['start'] == data.encode().index(b'"x"')
    check_position(data, item['position'], '"bad"')


def test_python_input_has_no_position():
    validator = simple_validator()
    with pytest.raises(Exception) as exc_info:
        validator.validate_python({'age': 'x'})
    assert 'position' not in exc_info.value.errors(include_position=True)[0]


def test_alias_field_name_differs_from_json_key():
    # with loc_by_alias=False the logical loc uses the field name while the JSON holds the alias;
    # the offending fragment must still be located via a containment search
    validator = SchemaValidator(
        s.model_fields_schema(
            {'my_field': s.model_field(s.int_schema(), validation_alias='myField')},
        ),
        config={'loc_by_alias': False},
    )
    data = '{\n  "other": 1,\n  "myField": "x"\n}'
    errors = positions(validator, data)
    assert errors[0]['loc'] == ('my_field',)
    check_position(data, errors[0]['position'], '"x"')


def test_union_branch_errors_point_at_each_branch_fragment():
    validator = SchemaValidator(s.list_schema(s.union_schema([s.int_schema(), s.list_schema(s.int_schema())])))
    data = '[1, [1, "bad"]]'
    errors = positions(validator, data)
    # the int branch rejects the whole inner array, the list branch rejects the bad element
    assert [e['loc'] for e in errors] == [(1, 'int'), (1, 'list[int]', 1)]
    check_position(data, errors[0]['position'], '[1, "bad"]')
    check_position(data, errors[1]['position'], '"bad"')


def test_duplicate_keys_point_to_last_occurrence():
    validator = SchemaValidator(s.typed_dict_schema({'a': s.typed_dict_field(s.int_schema())}))
    data = '{"a": 1, "a": "x"}'
    errors = positions(validator, data)
    assert len(errors) == 1
    check_position(data, errors[0]['position'], '"x"')
    assert errors[0]['position']['start'] == data.rindex('"x"')


def test_multibyte_characters_columns_count_code_points():
    validator = SchemaValidator(s.list_schema(s.int_schema()))
    data = '["你好", 1, "é"]'
    errors = positions(validator, data)
    first, second = errors
    check_position(data, first['position'], '"你好"')
    # the two CJK characters are 6 bytes wide but only 2 columns
    assert first['position']['start_column'] == 2
    assert first['position']['end_column'] == 6
    check_position(data, second['position'], '"é"')
    assert second['position']['start_column'] == 11


def test_unicode_escape_in_key_decoded_for_path_matching():
    validator = SchemaValidator(s.typed_dict_schema({'a.b': s.typed_dict_field(s.int_schema())}))
    data = '{"a\\u002eb": "x"}'
    errors = positions(validator, data)
    assert errors[0]['loc'] == ('a.b',)
    check_position(data, errors[0]['position'], '"x"')


def test_escaped_token_span_covers_the_whole_escape_sequence():
    validator = SchemaValidator(s.list_schema(s.int_schema()))
    # one code point written as an astral surrogate pair (12 raw characters)
    data = '["\\uD83D\\uDE00ab", "x"]'
    errors = positions(validator, data)
    check_position(data, errors[0]['position'], '"\\uD83D\\uDE00ab"')
    check_position(data, errors[1]['position'], '"x"')
    # the error input is the decoded value while the span covers the raw escape sequence
    assert errors[0]['input'] == '😀ab'


def test_multiline_line_and_column_coordinates():
    validator = SchemaValidator(s.typed_dict_schema({'a': s.typed_dict_field(s.int_schema())}))
    data = '{\n  "a":\n   "x"\n}'
    errors = positions(validator, data)
    position = errors[0]['position']
    assert (position['start_line'], position['start_column']) == (3, 4)
    assert (position['end_line'], position['end_column']) == (3, 7)
    check_position(data, position, '"x"')


def test_missing_field_points_at_closing_brace():
    validator = simple_validator()
    data = '{"name": "Bob"}'
    errors = positions(validator, data)
    assert [e['type'] for e in errors] == ['missing']
    position = errors[0]['position']
    # zero-width insertion point at the closing brace
    assert position['start'] == position['end'] == data.index('}')
    assert (position['start_line'], position['start_column']) == (1, len(data))


def test_parse_error_points_at_the_reported_byte():
    validator = simple_validator()
    data = '{"age": 12,}'
    errors = positions(validator, data)
    assert errors[0]['type'] == 'json_invalid'
    position = errors[0]['position']
    # jiter reports the unexpected closing brace after the trailing comma
    assert position['start'] == position['end'] == data.index('}')
    assert position['start_column'] == 12


def test_parse_error_at_eof():
    validator = simple_validator()
    errors = positions(validator, '[1')
    assert errors[0]['type'] == 'json_invalid'
    position = errors[0]['position']
    assert position['start'] == position['end'] == 2
    assert position['start_column'] == 3


def test_partial_on_truncated_array():
    validator = SchemaValidator(s.list_schema(s.int_schema()))
    # the trailing "3" is never parsed; "x" before it still has a position
    data = '[1, "x", 3'
    errors = positions(validator, data, allow_partial='on')
    assert errors[0]['loc'] == (1,)
    check_position(data, errors[0]['position'], '"x"')


def test_partial_on_unterminated_string_is_a_parse_error_at_eof():
    validator = SchemaValidator(s.str_schema())
    errors = positions(validator, '"abc', allow_partial='on')
    assert errors[0]['type'] == 'json_invalid'
    position = errors[0]['position']
    assert position['start'] == position['end'] == 4
    assert position['start_column'] == 5


def test_partial_trailing_strings_includes_unterminated_value():
    validator = SchemaValidator(s.typed_dict_schema({'a': s.typed_dict_field(s.int_schema())}))
    data = '{"a": "x'
    errors = positions(validator, data, allow_partial='trailing-strings')
    assert errors[0]['type'] == 'int_parsing'
    position = errors[0]['position']
    # the fragment runs to EOF, there is no closing quote in the document
    assert data[position['start'] :] == '"x'
    assert position['end'] == len(data)


def test_partial_on_missing_field_points_at_eof():
    validator = SchemaValidator(s.typed_dict_schema({'a': s.typed_dict_field(s.int_schema())}))
    data = '{"other": 1'
    errors = positions(validator, data, allow_partial='on')
    assert errors[0]['type'] == 'missing'
    position = errors[0]['position']
    assert position['start'] == position['end'] == len(data)


def test_json_output_round_trips_position():
    validator = simple_validator()
    with pytest.raises(Exception) as exc_info:
        validator.validate_json('{"age": "x"}')
    text = exc_info.value.json(include_position=True)
    parsed = json.loads(text)
    assert parsed[0]['position'] == exc_info.value.errors(include_position=True)[0]['position']
    # opt-out also applies to the JSON serializer
    assert 'position' not in json.loads(exc_info.value.json())[0]


def test_position_survives_pickling():
    validator = simple_validator()
    with pytest.raises(Exception) as exc_info:
        validator.validate_json('{"age": "x"}')
    original = exc_info.value.errors(include_position=True, include_url=False)
    restored = pickle.loads(pickle.dumps(exc_info.value)).errors(include_position=True, include_url=False)
    assert restored == original


def test_position_survives_from_exception_data():
    validator = simple_validator()
    with pytest.raises(Exception) as exc_info:
        validator.validate_json('{"age": "x"}')
    details = exc_info.value.errors(include_position=True, include_url=False)
    # details dicts can be copied and fed back through the constructor
    rebuilt = type(exc_info.value).from_exception_data('ValidationError', copy.deepcopy(details), 'json')
    assert rebuilt.errors(include_position=True, include_url=False) == details
