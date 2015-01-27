import numpy
import numpy.testing as nptest
import nose.tools as nt

import pygridgen


class _focusComponentMixin(object):
    _x = numpy.linspace(0, 1, num=10)
    _y = numpy.linspace(0, 1, num=10)
    x, y = numpy.meshgrid(_x, _y)

    pos = 0.25
    factor = 3
    reach = 0.2

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

    @nt.raises(ValueError)
    def test_call_bad_x_pos(self):
        self.focus([1.1], [0.5])

    @nt.raises(ValueError)
    def test_call_bad_x_neg(self):
        self.focus([-0.1], [0.5])

    @nt.raises(ValueError)
    def test_call_bad_y_pos(self):
        self.focus([0.5], [1.1])

    @nt.raises(ValueError)
    def test_call_bad_y_neg(self):
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
        nt.assert_true(hasattr(self.focus, self.pos_attr))
        nt.assert_equal(getattr(self.focus, self.pos_attr) , self.pos)

    def test_factor(self):
        nt.assert_true(hasattr(self.focus, 'factor'))
        nt.assert_equal(self.focus.factor, self.factor)

    def test_reach(self):
        nt.assert_true(hasattr(self.focus, self.reach_attr))
        nt.assert_equal(
            getattr(self.focus, self.reach_attr),
            self.reach
        )



class test__Focus_x(_focusComponentMixin):
    def setup(self):
        self.pos_attr = 'xo'
        self.reach_attr = 'Rx'
        self.focus = pygridgen.grid._Focus_x(
            self.pos, factor=self.factor, Rx=self.reach
        )

        self.known_focused_x = self.known_focused

        self.known_focused_y = self.y.copy()
        self.focused_x, self.focused_y = self.focus(self.x, self.y)


class test__Focus_y(_focusComponentMixin):
    def setup(self):
        self.pos_attr = 'yo'
        self.reach_attr = 'Ry'
        self.focus = pygridgen.grid._Focus_y(
            self.pos, factor=self.factor, Ry=self.reach
        )

        self.known_focused_x = self.x.copy()
        self.known_focused_y = self.known_focused.T

        self.focused_x, self.focused_y = self.focus(self.x, self.y)

