# -*- coding: utf-8 -*-
"""FrameHooks runs LIBRARY updaters before USER callbacks, on one context."""

import pytest

from hypertools.plot.animation_context import FrameHooks


def _hooks(**state):
    h = FrameHooks()
    h.record(frame=3, n_frames=10, **state)
    return h


def test_internal_updaters_run_before_user_callbacks():
    order = []
    h = _hooks()
    h.add(lambda ctx: order.append('user'))
    h.add_internal(lambda ctx: order.append('internal'))
    h.dispatch(figure=None, axes=None)
    # registered user-first on purpose: the PHASE decides, not the order
    assert order == ['internal', 'user']


def test_a_user_callback_sees_what_the_internal_updater_just_wrote():
    """The whole point: no user callback may observe a stale frame."""
    written = {}
    seen = []
    h = _hooks()
    h.add(lambda ctx: seen.append(written.get('frame')))
    h.add_internal(lambda ctx: written.__setitem__('frame', ctx.frame))
    h.dispatch(figure=None, axes=None)
    assert seen == [3]


def test_both_phases_share_one_frame_context():
    got = []
    h = _hooks()
    h.add(lambda ctx: got.append(ctx))
    h.add_internal(lambda ctx: got.append(ctx))
    h.dispatch(figure=None, axes=None)
    assert len(got) == 2 and got[0] is got[1]


def test_internal_updaters_run_with_no_user_callbacks_registered():
    """The guard that decides whether to dispatch must consider BOTH phases.

    An animated `predict=` with no `on_frame=` is the common case; if the
    early-return still asks only about user callbacks, the forecast never
    advances and nothing raises.
    """
    ran = []
    h = _hooks()
    h.add_internal(lambda ctx: ran.append(ctx.frame))
    h.dispatch(figure=None, axes=None)
    assert ran == [3]


def test_dispatch_is_a_no_op_when_nothing_is_registered():
    FrameHooks().dispatch(figure=None, axes=None)  # must not raise


def test_an_internal_updater_must_be_callable():
    with pytest.raises(TypeError, match='callable'):
        FrameHooks().add_internal(object())


def test_add_internal_returns_self_for_chaining():
    h = FrameHooks()
    assert h.add_internal(lambda ctx: None) is h


def test_an_exception_in_an_internal_updater_propagates():
    """Same contract as user callbacks: a broken hook is visible, never
    swallowed into a silently-wrong animation."""
    h = _hooks()
    h.add_internal(_raise)
    h.add(lambda ctx: pytest.fail('user callbacks must not run after a '
                                  'failed internal updater'))
    with pytest.raises(ValueError, match='boom'):
        h.dispatch(figure=None, axes=None)


def _raise(ctx):
    raise ValueError('boom')


def test_user_callbacks_still_run_in_registration_order():
    order = []
    h = _hooks()
    h.add(lambda ctx: order.append('a'))
    h.add(lambda ctx: order.append('b'))
    h.dispatch(figure=None, axes=None)
    assert order == ['a', 'b']
