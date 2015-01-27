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


class test_Focus(object):
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

        self.focus.add_focus_x(0.25, factor=3, Rx=0.1)
        self.focus.add_focus_x(0.75, factor=2, Rx=0.2)
        self.focus.add_focus_y(0.50, factor=2, Ry=0.3)

    def test__focuspoints(self):
        nt.assert_true(hasattr(self.focus, '_focuspoints'))
        nt.assert_true(isinstance(self.focus._focuspoints, list))
        nt.assert_equal(len(self.focus._focuspoints), 3)

    def test_add_focus_x(self):
        self.focus.add_focus_x(0.99, factor=3, Rx=0.1)
        nt.assert_true(len(self.focus._focuspoints), 4)
        nt.assert_true(isinstance(self.focus._focuspoints[-1], pygridgen.grid._Focus_x))
        nt.assert_equal(self.focus._focuspoints[-1].xo, 0.99)

    def test_add_focus_y(self):
        self.focus.add_focus_y(0.99, factor=3, Ry=0.1)
        nt.assert_true(len(self.focus._focuspoints), 4)
        nt.assert_true(isinstance(self.focus._focuspoints[-1], pygridgen.grid._Focus_y))
        nt.assert_equal(self.focus._focuspoints[-1].yo, 0.99)

    @nt.nottest
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
