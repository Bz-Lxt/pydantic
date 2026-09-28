import re
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import Final

import pytest
from typing_extensions import NotRequired, Required

from pydantic import BaseModel, ConfigDict, PydanticForbiddenQualifier, PydanticUserError, TypeAdapter, ValidationError
from pydantic.version import version_short


def test_user_error_url():
    with pytest.raises(
        PydanticUserError,
        check=lambda e: (
            str(e)
            == (
                'Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly\n\n'
                f'For further information visit https://errors.pydantic.dev/{version_short()}/u/base-model-instantiated'
            )
        ),
    ):
        BaseModel()


def test_forbidden_qualifier() -> None:
    with pytest.raises(PydanticForbiddenQualifier):

        class Model1(BaseModel):
            a: Required[int]

    with pytest.raises(PydanticForbiddenQualifier):

        class Model2(BaseModel):
            b: Final[NotRequired[str]] = 'test'


@pytest.mark.parametrize(
    'hide_input,input_str',
    ((False, 'type=greater_than, input_value=4, input_type=int'), (True, 'type=greater_than')),
)
def test_raise_validation_error_hide_input(hide_input, input_str):
    with pytest.raises(ValidationError, match=re.escape(f'Input should be greater than 5 [{input_str}]')):
        raise ValidationError.from_exception_data(
            'Foobar',
            [{'type': 'greater_than', 'loc': ('a', 2), 'input': 4, 'ctx': {'gt': 5}}],
            hide_input=hide_input,
        )


class FrozenModel(BaseModel):
    a: int
    model_config = ConfigDict(freeze_input_on_error=True)


def validate_frozen(input_value):
    with pytest.raises(ValidationError) as exc_info:
        FrozenModel.model_validate({'a': input_value})
    return exc_info.value.errors()[0]['input']


def test_error_input_mutable_by_default():
    # default behavior is unchanged: the error references the original object
    class MutableModel(BaseModel):
        a: int

    data = {'a': [1, 2]}
    with pytest.raises(ValidationError) as exc_info:
        MutableModel.model_validate(data)

    error = exc_info.value
    assert error.errors()[0]['input'] == [1, 2]
    # mutating the original container changes what the error shows
    data['a'].append(3)
    assert error.errors()[0]['input'] == [1, 2, 3]


def test_freeze_input_on_error_stable_after_mutation():
    data = {'a': [1, 2]}
    with pytest.raises(ValidationError) as exc_info:
        FrozenModel.model_validate(data)

    error = exc_info.value
    frozen_input = error.errors()[0]['input']
    assert frozen_input == (1, 2)
    assert isinstance(frozen_input, tuple)
    # mutations of the original input do not affect the frozen error
    data['a'].append(3)
    data['a'][0] = 99
    assert error.errors()[0]['input'] == (1, 2)


def test_freeze_input_containers_are_immutable():
    input_value = {'list': [1, [2, 3]], 'set': {1, 2}, 'tuple': (1, [2]), 'scalar': 'x', 'none': None}
    frozen_input = validate_frozen(input_value)
    assert isinstance(frozen_input, MappingProxyType)

    assert isinstance(frozen_input['list'], tuple)
    assert frozen_input['list'] == (1, (2, 3))
    assert isinstance(frozen_input['list'][1], tuple)

    assert isinstance(frozen_input['set'], frozenset)
    assert frozen_input['set'] == {1, 2}

    assert frozen_input['tuple'] == (1, (2,))
    assert frozen_input['scalar'] == 'x'
    assert frozen_input['none'] is None


def test_freeze_input_on_error_circular_reference_list():
    x = []
    x.append(x)
    frozen_input = validate_frozen(x)
    # the tuple references itself instead of looping forever
    assert isinstance(frozen_input, tuple)
    assert frozen_input[0] is frozen_input
    # rendering must terminate as well
    with pytest.raises(ValidationError) as exc_info:
        FrozenModel.model_validate({'a': x})
    assert str(exc_info.value)


def test_freeze_input_on_error_circular_reference_dict():
    d = {}
    d['self'] = d
    frozen_input = validate_frozen(d)
    assert isinstance(frozen_input, MappingProxyType)
    assert frozen_input['self'] is frozen_input
    with pytest.raises(ValidationError) as exc_info:
        FrozenModel.model_validate({'a': d})
    assert str(exc_info.value)


def test_freeze_input_on_error_shared_subobject():
    shared = [1, 2]
    container = {'x': shared, 'y': shared}
    frozen_input = validate_frozen(container)
    # sharing is preserved rather than duplicated
    assert frozen_input['x'] is frozen_input['y']
    # mutation of the shared original object is not visible
    shared.append(3)
    assert frozen_input['x'] == (1, 2)


def test_freeze_input_on_error_shared_across_line_errors():
    class MultiErrorModel(BaseModel):
        a: int
        b: int
        model_config = ConfigDict(freeze_input_on_error=True)

    shared = [1]
    with pytest.raises(ValidationError) as exc_info:
        MultiErrorModel.model_validate({'a': shared, 'b': shared})

    errors = exc_info.value.errors()
    assert errors[0]['input'] is errors[1]['input']
    assert errors[0]['input'] == (1,)


def test_freeze_input_on_error_str_and_json_stable():
    data = {'a': [1, 2]}
    with pytest.raises(ValidationError) as exc_info:
        FrozenModel.model_validate(data)

    error = exc_info.value
    str_before = str(error)
    json_before = error.json()
    errors_before = error.errors()

    data['a'].append(3)
    assert str(error) == str_before
    assert error.json() == json_before
    assert error.errors() == errors_before


def test_freeze_input_on_error_json_input():
    with pytest.raises(ValidationError) as exc_info:
        FrozenModel.model_validate_json('{"a": [1, 2]}')
    # JSON inputs are already owned, immutable snapshots
    assert exc_info.value.errors()[0]['input'] == [1, 2]


def test_freeze_input_on_error_type_adapter():
    adapter = TypeAdapter(int, config=ConfigDict(freeze_input_on_error=True))
    with pytest.raises(ValidationError) as exc_info:
        adapter.validate_python([1, 2])
    assert exc_info.value.errors()[0]['input'] == (1, 2)


def test_freeze_input_on_error_concurrent_reads():
    data = {'a': [1, 2]}
    with pytest.raises(ValidationError) as exc_info:
        FrozenModel.model_validate(data)

    error = exc_info.value
    data['a'].append(3)

    def read_error():
        return str(error), error.json(), error.errors()

    expected = read_error()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: read_error(), range(40)))
    assert all(result == expected for result in results)


def test_freeze_input_on_error_pickle_roundtrip():
    import pickle

    data = {'a': [1, 2]}
    with pytest.raises(ValidationError) as exc_info:
        FrozenModel.model_validate(data)

    error = pickle.loads(pickle.dumps(exc_info.value))
    data['a'].append(3)
    assert error.errors()[0]['input'] == (1, 2)
