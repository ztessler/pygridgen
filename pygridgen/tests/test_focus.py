import numpy

import numpy.testing as nptest
import pytest

import pygridgen


class _focusPointMixin(object):
    _x = numpy.linspace(0, 1, num=10)
    _y = numpy.linspace(0, 1, num=10)
    x, y = numpy.meshgrid(_x, _y)

    pos = 0.25
    factor = 3
    reach = 0.2
    pos_attr = 'pos'
    reach_attr = 'extent'

    known_focused = numpy.array([
        [0., 0.106, 0.17, 0.222, 0.307, 0.43, 0.569, 0.712, 0.856, 1.],
        [0., 0.106, 0.17, 0.222, 0.307, 0.43, 0.569, 0.712, 0.856, 1.],
        [0., 0.106, 0.17, 0.222, 0.307, 0.43, 0.569, 0.712, 0.856, 1.],
        [0., 0.106, 0.17, 0.222, 0.307, 0.43, 0.569, 0.712, 0.856, 1.],
        [0., 0.106, 0.17, 0.222, 0.307, 0.43, 0.569, 0.712, 0.856, 1.],
        [0., 0.106, 0.17, 0.222, 0.307, 0.43, 0.569, 0.712, 0.856, 1.],
        [0., 0.106, 0.17, 0.222, 0.307, 0.43, 0.569, 0.712, 0.856, 1.],
        [0., 0.106, 0.17, 0.222, 0.307, 0.43, 0.569, 0.712, 0.856, 1.],
        [0., 0.106, 0.17, 0.222, 0.307, 0.43, 0.569, 0.712, 0.856, 1.],
        [0., 0.106, 0.17, 0.222, 0.307, 0.43, 0.569, 0.712, 0.856, 1.]
    ])

    def setup(self):
        self.main_setup()
        self.focus = pygridgen.grid._FocusPoint(
            self.pos, self.axis, self.factor, self.reach
        )
        self.focused_x, self.focused_y = self.focus(self.x, self.y)

    def test_call_bad_x_pos(self):
        with pytest.raises(ValueError):
            self.focus([1.1], [0.5])

    def test_call_bad_x_neg(self):
        with pytest.raises(ValueError):
            self.focus([-0.1], [0.5])

    def test_call_bad_y_pos(self):
        with pytest.raises(ValueError):
            self.focus([0.5], [1.1])

    def test_call_bad_y_neg(self):
        with pytest.raises(ValueError):
            self.focus([0.5], [-0.1])

    def test_focused_x(self):
        nptest.assert_array_almost_equal(
            self.focused_x,
            self.known_focused_x,
            decimal=3
        )

    def test_focused_y(self):
        nptest.assert_array_almost_equal(
            self.focused_y,
            self.known_focused_y,
            decimal=3
        )

    def test_pos(self):
        assert getattr(self.focus, self.pos_attr) == self.pos

    def test_factor(self):
        assert self.focus.factor == self.factor

    def test_reach(self):
        assert getattr(self.focus, self.reach_attr) == self.reach

    def test_badaxis(self):
        with pytest.raises(ValueError):
            focus = pygridgen.grid._FocusPoint(0.1, 'xjunk', 2, 0.25)

    def test_position_positive(self):
        with pytest.raises(ValueError):
            focus = pygridgen.grid._FocusPoint(1.1, self.axis, 2, 0.25)

    def test_position_negative(self):
        with pytest.raises(ValueError):
            focus = pygridgen.grid._FocusPoint(-0.1, self.axis, 2, 0.25)


class Test__Focus_x(_focusPointMixin):
    def main_setup(self):
        self.axis = 'x'
        self.focus = pygridgen.grid._FocusPoint(
            self.pos, self.axis, self.factor, self.reach
        )
        self.known_focused_x = self.known_focused
        self.known_focused_y = self.y.copy()
        self.focused_x, self.focused_y = self.focus(self.x, self.y)


class Test__Focus_y(_focusPointMixin):
    def main_setup(self):
        self.axis = 'y'
        self.pos_attr = 'pos'
        self.reach_attr = 'extent'

        self.known_focused_x = self.x.copy()
        self.known_focused_y = self.known_focused.T


class Test_Focus(object):
    def setup(self):
        self.focus = pygridgen.Focus()
        self.known_focused_x = numpy.array([
            [0., 0.148, 0.248, 0.313, 0.446, 0.59, 0.711, 0.796, 0.881, 1.],
            [0., 0.148, 0.248, 0.313, 0.446, 0.59, 0.711, 0.796, 0.881, 1.],
            [0., 0.148, 0.248, 0.313, 0.446, 0.59, 0.711, 0.796, 0.881, 1.],
            [0., 0.148, 0.248, 0.313, 0.446, 0.59, 0.711, 0.796, 0.881, 1.],
            [0., 0.148, 0.248, 0.313, 0.446, 0.59, 0.711, 0.796, 0.881, 1.],
            [0., 0.148, 0.248, 0.313, 0.446, 0.59, 0.711, 0.796, 0.881, 1.],
            [0., 0.148, 0.248, 0.313, 0.446, 0.59, 0.711, 0.796, 0.881, 1.],
            [0., 0.148, 0.248, 0.313, 0.446, 0.59, 0.711, 0.796, 0.881, 1.],
            [0., 0.148, 0.248, 0.313, 0.446, 0.59, 0.711, 0.796, 0.881, 1.],
            [0., 0.148, 0.248, 0.313, 0.446, 0.59, 0.711, 0.796, 0.881, 1.]
        ])

        self.known_focused_y = numpy.array([
            [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
            [0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142],
            [0.27 , 0.27 , 0.27 , 0.27 , 0.27 , 0.27 , 0.27 , 0.27 , 0.27 , 0.27 ],
            [0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377],
            [0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462],
            [0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538, 0.538],
            [0.623, 0.623, 0.623, 0.623, 0.623, 0.623, 0.623, 0.623, 0.623, 0.623],
            [0.73 , 0.73 , 0.73 , 0.73 , 0.73 , 0.73 , 0.73 , 0.73 , 0.73 , 0.73 ],
            [0.858, 0.858, 0.858, 0.858, 0.858, 0.858, 0.858, 0.858, 0.858, 0.858],
            [1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   ]
        ])

        self.focus.add_focus(0.25, 'x', factor=3, extent=0.1)
        self.focus.add_focus(0.75, 'x', factor=2, extent=0.2)
        self.focus.add_focus(0.50, 'y', factor=2, extent=0.3)

    def test__focuspoints(self):
        assert isinstance(self.focus._focuspoints, list)
        assert len(self.focus._focuspoints) == 3

    def test_add_focus_x(self):
        self.focus.add_focus(0.99, 'x', factor=3, extent=0.1)
        assert len(self.focus._focuspoints), 4
        assert isinstance(self.focus._focuspoints[-1], pygridgen.grid._FocusPoint)
        assert self.focus._focuspoints[-1].pos == 0.99
        assert self.focus._focuspoints[-1].axis == 'x'

    def test_add_focus_y(self):
        self.focus.add_focus(0.99, 'y', factor=3, extent=0.1)
        assert len(self.focus._focuspoints), 4
        assert isinstance(self.focus._focuspoints[-1], pygridgen.grid._FocusPoint)
        assert self.focus._focuspoints[-1].pos == 0.99
        assert self.focus._focuspoints[-1].axis == 'y'

    def do_call(self):
        _x = numpy.linspace(0, 1, num=10)
        _y = numpy.linspace(0, 1, num=10)
        x, y = numpy.meshgrid(_x, _y)
        return self.focus(x, y)

    def test_focused_x(self):
        xf, yf = self.do_call()
        nptest.assert_array_almost_equal(xf, self.known_focused_x, decimal=3)

    def test_focused_y(self):
        xf, yf = self.do_call()
        nptest.assert_array_almost_equal(yf, self.known_focused_y, decimal=3)
