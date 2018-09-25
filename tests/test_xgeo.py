#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the xgeo module.
"""
import pytest

from xgeo import xgeo


def test_something():
    assert True


def test_with_error():
    with pytest.raises(ValueError):
        # Do something that raises a ValueError
        raise(ValueError)


# Fixture example
@pytest.fixture
def an_object():
    return {}


def test_xgeo(an_object):
    assert an_object == {}
