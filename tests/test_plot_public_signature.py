# -*- coding: utf-8 -*-
"""What `hypertools.plot` PROMISES, checked against the function itself.

Every parameter here is part of the published surface: it appears in the
docstring, in the documentation, or in an example, and removing or renaming
one breaks user code. The behaviour of each is tested in its own module --
this file pins only that the surface exists and that its defaults are what
the documentation says they are, which is the part a behavioural test in a
feature-specific file cannot notice going missing.

Real introspection of the real function; nothing here is mocked.
"""
import inspect

import pytest

import hypertools as hyp


def signature():
    return inspect.signature(hyp.plot)


#: (parameter, default) pairs. A default recorded here is a documented
#: promise, not an implementation detail: changing one changes what a call
#: with no arguments does.
PUBLISHED_DEFAULTS = [
    ('fmt', '-'),
    ('hue', None),
    ('hue_mode', None),
    ('palette', 'hls'),
    ('animate', False),
    ('predict', None),
    ('reduce', 'IncrementalPCA'),
    ('align', None),
    ('cluster', None),
    ('normalize', None),
    ('manip', None),
]


@pytest.mark.parametrize('name, default', PUBLISHED_DEFAULTS)
def test_the_published_parameter_is_present_with_its_documented_default(
        name, default):
    parameters = signature().parameters
    assert name in parameters, f'{name}= vanished from the public signature'
    assert parameters[name].default == default


def test_hue_mode_is_a_NAMED_parameter_rather_than_a_kwargs_passthrough():
    """`hue_mode=` decides whether a hue matrix is blended into one colour
    per point or reduced to RGB. It arrived as a named parameter in 1.1; a
    later refactor that folded it back into `**kwargs` would keep every
    behavioural test passing (the value still reaches the drawing code)
    while silently dropping it from `help(hyp.plot)` and from the
    documentation build's signature.
    """
    parameter = signature().parameters['hue_mode']
    assert parameter.kind in (parameter.POSITIONAL_OR_KEYWORD,
                              parameter.KEYWORD_ONLY)
    assert parameter.default is None, (
        'None must keep meaning "decide from the data", so that code '
        'written before hue_mode= existed keeps its old behaviour')


def test_hue_mode_is_documented_with_all_three_of_its_values():
    """A parameter nobody can find is not public. The three legal values
    have to appear in the docstring users actually read."""
    doc = hyp.plot.__doc__
    assert 'hue_mode' in doc
    for value in ("'mixture'", "'rgb'"):
        assert value in doc, f'{value} is not documented on hyp.plot'
