"""
Test suite for deriv_poly Python bindings

Tests BPoly, CPoly, LegPoly, HPoly, BsPoly, LagPoly and their list variants.
"""

import pytest
import numpy as np


def test_import():
    """Test that the module can be imported."""
    import deriv_poly as dp
    assert hasattr(dp, 'BPoly')
    assert hasattr(dp, 'CPoly')
    assert hasattr(dp, 'LegPoly')
    assert hasattr(dp, 'HPoly')
    assert hasattr(dp, 'BsPoly')
    assert hasattr(dp, 'LagPoly')
    assert hasattr(dp, 'BPolyList')
    assert hasattr(dp, 'HPolyList')
    assert hasattr(dp, 'HermiteKind')


class TestBPoly:
    """Tests for BPoly class."""

    def test_construction(self):
        """Test basic construction from coefficients."""
        import deriv_poly as dp
        # Linear polynomial on [0, 1]: f(t) = (1-t)*0 + t*1 = t, so f(x) = x
        coeffs = [[0], [1]]  # [degree+1][num_intervals]
        breaks = [0, 1]
        p = dp.BPoly(coeffs, breaks)
        assert p.degree == 1
        assert p.num_intervals == 1

    def test_evaluation_scalar(self):
        """Test scalar evaluation."""
        import deriv_poly as dp
        # Constant polynomial f(x) = 2
        p = dp.BPoly([[2]], [0, 1])
        assert abs(p(0.5) - 2.0) < 1e-10

    def test_evaluation_array(self):
        """Test array evaluation."""
        import deriv_poly as dp
        # Linear f(x) = x on [0,1]
        p = dp.BPoly([[0], [1]], [0, 1])
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = p(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == (5,)
        np.testing.assert_allclose(y, x, atol=1e-10)

    def test_evaluation_2d_array(self):
        """Test 2D array evaluation preserves shape."""
        import deriv_poly as dp
        p = dp.BPoly([[0], [1]], [0, 1])
        x = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = p(x)
        assert y.shape == (2, 2)
        np.testing.assert_allclose(y, x, atol=1e-10)

    def test_from_derivatives(self):
        """Test from_derivatives construction."""
        import deriv_poly as dp
        # Hermite cubic: f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
        xi = [0, 1]
        yi = [[0, 1], [1, -1]]
        p = dp.BPoly.from_derivatives(xi, yi)
        assert p.degree == 3
        assert abs(p(0.0) - 0.0) < 1e-10
        assert abs(p(1.0) - 1.0) < 1e-10

    def test_derivative(self):
        """Test derivative computation."""
        import deriv_poly as dp
        # f(x) = x^2 -> f'(x) = 2x
        # Bernstein coeffs for x^2 on [0,1]: [0, 0, 1]
        p = dp.BPoly([[0], [0], [1]], [0, 1])
        dp_poly = p.derivative()
        # f'(0.5) should be 1.0
        assert abs(dp_poly(0.5) - 1.0) < 1e-10

    def test_integration(self):
        """Test definite integration."""
        import deriv_poly as dp
        # f(x) = 1 -> integral from 0 to 1 = 1
        p = dp.BPoly([[1]], [0, 1])
        result = p.integrate(0, 1)
        assert abs(result - 1.0) < 1e-10

    def test_derivative_nu_parameter(self):
        """Test derivative evaluation via nu parameter."""
        import deriv_poly as dp
        p = dp.BPoly([[0], [0], [1]], [0, 1])  # x^2
        # f(0.5, nu=1) = f'(0.5) = 2*0.5 = 1.0
        assert abs(p(0.5, nu=1) - 1.0) < 1e-10


class TestBPolyList:
    """Tests for BPolyList class."""

    def test_construction_empty(self):
        """Test empty list construction."""
        import deriv_poly as dp
        pl = dp.BPolyList()
        assert len(pl) == 0

    def test_construction_from_list(self):
        """Test construction from Python list."""
        import deriv_poly as dp
        p1 = dp.BPoly([[1]], [0, 1])
        p2 = dp.BPoly([[2]], [0, 1])
        pl = dp.BPolyList([p1, p2])
        assert len(pl) == 2

    def test_append(self):
        """Test append method."""
        import deriv_poly as dp
        pl = dp.BPolyList()
        p1 = dp.BPoly([[1]], [0, 1])
        pl.append(p1)
        assert len(pl) == 1

    def test_extend(self):
        """Test extend method."""
        import deriv_poly as dp
        pl = dp.BPolyList()
        p1 = dp.BPoly([[1]], [0, 1])
        p2 = dp.BPoly([[2]], [0, 1])
        pl.extend([p1, p2])
        assert len(pl) == 2

    def test_evaluation_scalar(self):
        """Test list evaluation at scalar returns (N,) array."""
        import deriv_poly as dp
        p1 = dp.BPoly([[1]], [0, 1])  # f(x) = 1
        p2 = dp.BPoly([[2]], [0, 1])  # f(x) = 2
        p3 = dp.BPoly([[3]], [0, 1])  # f(x) = 3
        pl = dp.BPolyList([p1, p2, p3])

        y = pl(0.5)
        assert isinstance(y, np.ndarray)
        assert y.shape == (3,)
        np.testing.assert_allclose(y, [1.0, 2.0, 3.0], atol=1e-10)

    def test_evaluation_1d_array(self):
        """Test list evaluation at 1D array returns (N, M) array."""
        import deriv_poly as dp
        p1 = dp.BPoly([[1]], [0, 1])
        p2 = dp.BPoly([[2]], [0, 1])
        pl = dp.BPolyList([p1, p2])

        x = np.array([0.1, 0.2, 0.3, 0.4])
        y = pl(x)
        assert y.shape == (2, 4)
        np.testing.assert_allclose(y[0], [1.0, 1.0, 1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(y[1], [2.0, 2.0, 2.0, 2.0], atol=1e-10)

    def test_evaluation_2d_array(self):
        """Test list evaluation at 2D array returns (N, M, K) array."""
        import deriv_poly as dp
        p1 = dp.BPoly([[1]], [0, 1])
        p2 = dp.BPoly([[2]], [0, 1])
        pl = dp.BPolyList([p1, p2])

        x = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # shape (3, 2)
        y = pl(x)
        assert y.shape == (2, 3, 2)

    def test_evaluation_with_derivative(self):
        """Test list evaluation with nu parameter."""
        import deriv_poly as dp
        # f(x) = x, f'(x) = 1
        p1 = dp.BPoly([[0], [1]], [0, 1])
        # f(x) = x^2, f'(x) = 2x
        p2 = dp.BPoly([[0], [0], [1]], [0, 1])
        pl = dp.BPolyList([p1, p2])

        y = pl(0.5, nu=1)
        assert y.shape == (2,)
        np.testing.assert_allclose(y, [1.0, 1.0], atol=1e-10)

    def test_integrate(self):
        """Test list integration."""
        import deriv_poly as dp
        p1 = dp.BPoly([[1]], [0, 1])  # integral = 1
        p2 = dp.BPoly([[2]], [0, 1])  # integral = 2
        pl = dp.BPolyList([p1, p2])

        result = pl.integrate(0, 1)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-10)

    def test_derivative_list(self):
        """Test derivative returns new list."""
        import deriv_poly as dp
        p1 = dp.BPoly([[0], [1]], [0, 1])  # f(x) = x
        p2 = dp.BPoly([[0], [0], [1]], [0, 1])  # f(x) = x^2
        pl = dp.BPolyList([p1, p2])

        dpl = pl.derivative()
        assert len(dpl) == 2
        # f'(x) = 1 for p1
        assert abs(dpl[0](0.5) - 1.0) < 1e-10

    def test_setitem_basic(self):
        """Test __setitem__ replaces polynomial correctly."""
        import deriv_poly as dp
        p1 = dp.BPoly([[1]], [0, 1])  # f(x) = 1
        p2 = dp.BPoly([[2]], [0, 1])  # f(x) = 2
        p3 = dp.BPoly([[3]], [0, 1])  # f(x) = 3
        pl = dp.BPolyList([p1, p2, p3])

        # Replace middle polynomial
        p_new = dp.BPoly([[5]], [0, 1])  # f(x) = 5
        pl[1] = p_new

        assert len(pl) == 3
        y = pl(0.5)
        np.testing.assert_allclose(y, [1.0, 5.0, 3.0], atol=1e-10)

    def test_setitem_first(self):
        """Test __setitem__ at first index."""
        import deriv_poly as dp
        p1 = dp.BPoly([[1]], [0, 1])
        p2 = dp.BPoly([[2]], [0, 1])
        pl = dp.BPolyList([p1, p2])

        p_new = dp.BPoly([[10]], [0, 1])
        pl[0] = p_new

        y = pl(0.5)
        np.testing.assert_allclose(y, [10.0, 2.0], atol=1e-10)

    def test_setitem_last(self):
        """Test __setitem__ at last index."""
        import deriv_poly as dp
        p1 = dp.BPoly([[1]], [0, 1])
        p2 = dp.BPoly([[2]], [0, 1])
        pl = dp.BPolyList([p1, p2])

        p_new = dp.BPoly([[20]], [0, 1])
        pl[1] = p_new

        y = pl(0.5)
        np.testing.assert_allclose(y, [1.0, 20.0], atol=1e-10)

    def test_setitem_out_of_range(self):
        """Test __setitem__ raises on invalid index."""
        import deriv_poly as dp
        p1 = dp.BPoly([[1]], [0, 1])
        pl = dp.BPolyList([p1])

        p_new = dp.BPoly([[5]], [0, 1])
        with pytest.raises(IndexError):
            pl[5] = p_new

    def test_setitem_preserves_other_elements(self):
        """Test __setitem__ doesn't affect other elements."""
        import deriv_poly as dp
        # Create polynomials with different degrees
        p1 = dp.BPoly([[0], [1]], [0, 1])      # linear: f(x) = x
        p2 = dp.BPoly([[0], [0], [1]], [0, 1])  # quadratic: f(x) = x^2
        p3 = dp.BPoly([[1]], [0, 1])            # constant: f(x) = 1
        pl = dp.BPolyList([p1, p2, p3])

        # Replace middle
        p_new = dp.BPoly([[7]], [0, 1])  # constant: f(x) = 7
        pl[1] = p_new

        # Check all values at x=0.5
        y = pl(0.5)
        np.testing.assert_allclose(y, [0.5, 7.0, 1.0], atol=1e-10)


class TestCPoly:
    """Tests for CPoly class."""

    def test_construction(self):
        """Test CPoly construction."""
        import deriv_poly as dp
        # Chebyshev: c_0 = 1 means constant 1
        p = dp.CPoly([[1]], [0, 1])
        assert p.degree == 0

    def test_evaluation(self):
        """Test CPoly evaluation."""
        import deriv_poly as dp
        p = dp.CPoly([[2]], [0, 1])  # constant 2
        assert abs(p(0.5) - 2.0) < 1e-10


class TestLegPoly:
    """Tests for LegPoly class."""

    def test_construction(self):
        """Test LegPoly construction."""
        import deriv_poly as dp
        p = dp.LegPoly([[1]], [0, 1])
        assert p.degree == 0

    def test_evaluation(self):
        """Test LegPoly evaluation."""
        import deriv_poly as dp
        p = dp.LegPoly([[2]], [0, 1])
        assert abs(p(0.5) - 2.0) < 1e-10


class TestBsPoly:
    """Tests for BsPoly class."""

    def test_construction(self):
        """Test BsPoly construction."""
        import deriv_poly as dp
        p = dp.BsPoly([[1]], [0, 1])
        assert p.degree == 0

    def test_evaluation(self):
        """Test BsPoly evaluation."""
        import deriv_poly as dp
        p = dp.BsPoly([[2]], [0, 1])
        assert abs(p(0.5) - 2.0) < 1e-10


class TestLagPoly:
    """Tests for LagPoly class."""

    def test_from_derivatives(self):
        """Test LagPoly from_derivatives."""
        import deriv_poly as dp
        xi = [0, 1]
        yi = [[0], [1]]  # f(0) = 0, f(1) = 1
        p = dp.LagPoly.from_derivatives(xi, yi)
        assert abs(p(0.0) - 0.0) < 1e-10
        assert abs(p(1.0) - 1.0) < 1e-10


class TestHPoly:
    """Tests for HPoly class."""

    def test_construction_physicist(self):
        """Test HPoly construction with Physicist's variant (default)."""
        import deriv_poly as dp
        # Constant polynomial in Hermite basis
        p = dp.HPoly([[1]], [0, 1])
        assert p.degree == 0
        assert p.kind == dp.HermiteKind.Physicist

    def test_construction_probabilist(self):
        """Test HPoly construction with Probabilist's variant."""
        import deriv_poly as dp
        p = dp.HPoly([[1]], [0, 1], kind=dp.HermiteKind.Probabilist)
        assert p.degree == 0
        assert p.kind == dp.HermiteKind.Probabilist

    def test_evaluation_constant(self):
        """Test HPoly evaluation of constant polynomial."""
        import deriv_poly as dp
        # Both variants: H_0(x) = He_0(x) = 1, so c_0 = 2 gives f(x) = 2
        p_phys = dp.HPoly([[2]], [0, 1], kind=dp.HermiteKind.Physicist)
        p_prob = dp.HPoly([[2]], [0, 1], kind=dp.HermiteKind.Probabilist)
        assert abs(p_phys(0.5) - 2.0) < 1e-10
        assert abs(p_prob(0.5) - 2.0) < 1e-10

    def test_evaluation_array(self):
        """Test HPoly array evaluation."""
        import deriv_poly as dp
        p = dp.HPoly([[2]], [0, 1])
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = p(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == (5,)
        np.testing.assert_allclose(y, [2.0, 2.0, 2.0, 2.0, 2.0], atol=1e-10)

    def test_from_derivatives_physicist(self):
        """Test HPoly.from_derivatives with Physicist's variant."""
        import deriv_poly as dp
        # f(0) = 0, f'(0) = 1, f(1) = 1, f'(1) = -1
        xi = [0, 1]
        yi = [[0, 1], [1, -1]]
        p = dp.HPoly.from_derivatives(xi, yi, kind=dp.HermiteKind.Physicist)
        assert p.degree == 3
        assert abs(p(0.0) - 0.0) < 1e-10
        assert abs(p(1.0) - 1.0) < 1e-10

    def test_from_derivatives_probabilist(self):
        """Test HPoly.from_derivatives with Probabilist's variant."""
        import deriv_poly as dp
        xi = [0, 1]
        yi = [[0, 1], [1, -1]]
        p = dp.HPoly.from_derivatives(xi, yi, kind=dp.HermiteKind.Probabilist)
        assert p.degree == 3
        assert abs(p(0.0) - 0.0) < 1e-10
        assert abs(p(1.0) - 1.0) < 1e-10

    def test_derivative(self):
        """Test HPoly derivative computation."""
        import deriv_poly as dp
        # Create polynomial via from_derivatives
        p = dp.HPoly.from_derivatives([0, 1], [[0, 1], [1, 0]])
        dp_poly = p.derivative()
        # Check derivative values
        assert abs(dp_poly(0.0) - 1.0) < 1e-10

    def test_derivative_nu_parameter(self):
        """Test HPoly derivative evaluation via nu parameter."""
        import deriv_poly as dp
        p = dp.HPoly.from_derivatives([0, 1], [[0, 1], [1, 0]])
        # f'(0) should be 1.0 (specified in from_derivatives)
        assert abs(p(0.0, nu=1) - 1.0) < 1e-10

    def test_integration(self):
        """Test HPoly integration."""
        import deriv_poly as dp
        # Constant polynomial f(x) = 1: integral from 0 to 1 = 1
        p = dp.HPoly([[1]], [0, 1])
        result = p.integrate(0, 1)
        assert abs(result - 1.0) < 1e-10

    def test_kind_property(self):
        """Test HPoly.kind property."""
        import deriv_poly as dp
        p1 = dp.HPoly([[1]], [0, 1], kind=dp.HermiteKind.Physicist)
        p2 = dp.HPoly([[1]], [0, 1], kind=dp.HermiteKind.Probabilist)
        assert p1.kind == dp.HermiteKind.Physicist
        assert p2.kind == dp.HermiteKind.Probabilist


class TestHPolyList:
    """Tests for HPolyList class."""

    def test_construction_empty(self):
        """Test empty HPolyList construction."""
        import deriv_poly as dp
        pl = dp.HPolyList()
        assert len(pl) == 0

    def test_construction_from_list(self):
        """Test HPolyList construction from list."""
        import deriv_poly as dp
        p1 = dp.HPoly([[1]], [0, 1])
        p2 = dp.HPoly([[2]], [0, 1])
        pl = dp.HPolyList([p1, p2])
        assert len(pl) == 2

    def test_append(self):
        """Test HPolyList append."""
        import deriv_poly as dp
        pl = dp.HPolyList()
        p1 = dp.HPoly([[1]], [0, 1])
        pl.append(p1)
        assert len(pl) == 1

    def test_evaluation_scalar(self):
        """Test HPolyList evaluation at scalar."""
        import deriv_poly as dp
        p1 = dp.HPoly([[1]], [0, 1])
        p2 = dp.HPoly([[2]], [0, 1])
        p3 = dp.HPoly([[3]], [0, 1])
        pl = dp.HPolyList([p1, p2, p3])

        y = pl(0.5)
        assert isinstance(y, np.ndarray)
        assert y.shape == (3,)
        np.testing.assert_allclose(y, [1.0, 2.0, 3.0], atol=1e-10)

    def test_evaluation_1d_array(self):
        """Test HPolyList evaluation at 1D array."""
        import deriv_poly as dp
        p1 = dp.HPoly([[1]], [0, 1])
        p2 = dp.HPoly([[2]], [0, 1])
        pl = dp.HPolyList([p1, p2])

        x = np.array([0.1, 0.2, 0.3, 0.4])
        y = pl(x)
        assert y.shape == (2, 4)
        np.testing.assert_allclose(y[0], [1.0, 1.0, 1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(y[1], [2.0, 2.0, 2.0, 2.0], atol=1e-10)

    def test_integrate(self):
        """Test HPolyList integration."""
        import deriv_poly as dp
        p1 = dp.HPoly([[1]], [0, 1])
        p2 = dp.HPoly([[2]], [0, 1])
        pl = dp.HPolyList([p1, p2])

        result = pl.integrate(0, 1)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-10)

    def test_setitem(self):
        """Test HPolyList __setitem__."""
        import deriv_poly as dp
        p1 = dp.HPoly([[1]], [0, 1])
        p2 = dp.HPoly([[2]], [0, 1])
        pl = dp.HPolyList([p1, p2])

        p_new = dp.HPoly([[5]], [0, 1])
        pl[0] = p_new

        y = pl(0.5)
        np.testing.assert_allclose(y, [5.0, 2.0], atol=1e-10)

    def test_mixed_kinds_in_list(self):
        """Test HPolyList with mixed Physicist/Probabilist polynomials."""
        import deriv_poly as dp
        p_phys = dp.HPoly([[1]], [0, 1], kind=dp.HermiteKind.Physicist)
        p_prob = dp.HPoly([[1]], [0, 1], kind=dp.HermiteKind.Probabilist)
        pl = dp.HPolyList([p_phys, p_prob])

        assert len(pl) == 2
        y = pl(0.5)
        # Both should evaluate to 1 for constant polynomial
        np.testing.assert_allclose(y, [1.0, 1.0], atol=1e-10)


class TestPolyListSetItem:
    """Test __setitem__ across all PolyList types."""

    def test_cpoly_list_setitem(self):
        """Test CPolyList __setitem__."""
        import deriv_poly as dp
        p1 = dp.CPoly([[1]], [0, 1])
        p2 = dp.CPoly([[2]], [0, 1])
        pl = dp.CPolyList([p1, p2])

        p_new = dp.CPoly([[5]], [0, 1])
        pl[0] = p_new

        y = pl(0.5)
        np.testing.assert_allclose(y, [5.0, 2.0], atol=1e-10)

    def test_lpoly_list_setitem(self):
        """Test LegPolyList __setitem__."""
        import deriv_poly as dp
        p1 = dp.LegPoly([[1]], [0, 1])
        p2 = dp.LegPoly([[2]], [0, 1])
        pl = dp.LegPolyList([p1, p2])

        p_new = dp.LegPoly([[5]], [0, 1])
        pl[1] = p_new

        y = pl(0.5)
        np.testing.assert_allclose(y, [1.0, 5.0], atol=1e-10)

    def test_bspoly_list_setitem(self):
        """Test BsPolyList __setitem__."""
        import deriv_poly as dp
        p1 = dp.BsPoly([[1]], [0, 1])
        p2 = dp.BsPoly([[2]], [0, 1])
        pl = dp.BsPolyList([p1, p2])

        p_new = dp.BsPoly([[5]], [0, 1])
        pl[0] = p_new

        y = pl(0.5)
        np.testing.assert_allclose(y, [5.0, 2.0], atol=1e-10)

    def test_lagpoly_list_setitem(self):
        """Test LagPolyList __setitem__."""
        import deriv_poly as dp
        # LagPoly needs from_derivatives
        p1 = dp.LagPoly.from_derivatives([0, 1], [[1], [1]])  # constant 1
        p2 = dp.LagPoly.from_derivatives([0, 1], [[2], [2]])  # constant 2
        pl = dp.LagPolyList([p1, p2])

        p_new = dp.LagPoly.from_derivatives([0, 1], [[5], [5]])  # constant 5
        pl[1] = p_new

        y = pl(0.5)
        np.testing.assert_allclose(y, [1.0, 5.0], atol=1e-10)

    def test_hpoly_list_setitem(self):
        """Test HPolyList __setitem__."""
        import deriv_poly as dp
        p1 = dp.HPoly([[1]], [0, 1])
        p2 = dp.HPoly([[2]], [0, 1])
        pl = dp.HPolyList([p1, p2])

        p_new = dp.HPoly([[5]], [0, 1])
        pl[0] = p_new

        y = pl(0.5)
        np.testing.assert_allclose(y, [5.0, 2.0], atol=1e-10)


class TestFlexibleInput:
    """Tests for flexible input handling (numpy arrays, tuples, lists)."""

    def test_bpoly_construction_from_numpy(self):
        """Test BPoly construction with numpy arrays."""
        import deriv_poly as dp
        coeffs = np.array([[0.0], [1.0]])
        breaks = np.array([0.0, 1.0])
        p = dp.BPoly(coeffs, breaks)
        assert p.degree == 1
        np.testing.assert_allclose(p(0.5), 0.5, atol=1e-10)

    def test_bpoly_construction_from_tuple(self):
        """Test BPoly construction with tuples."""
        import deriv_poly as dp
        coeffs = ((0.0,), (1.0,))
        breaks = (0.0, 1.0)
        p = dp.BPoly(coeffs, breaks)
        assert p.degree == 1
        np.testing.assert_allclose(p(0.5), 0.5, atol=1e-10)

    def test_bpoly_construction_from_mixed(self):
        """Test BPoly construction with mixed list/tuple."""
        import deriv_poly as dp
        coeffs = [(0.0,), (1.0,)]  # list of tuples
        breaks = (0.0, 1.0)  # tuple
        p = dp.BPoly(coeffs, breaks)
        assert p.degree == 1

    def test_bpoly_from_derivatives_numpy(self):
        """Test BPoly.from_derivatives with numpy arrays."""
        import deriv_poly as dp
        xi = np.array([0.0, 1.0])
        yi = np.array([[0.0, 1.0], [1.0, -1.0]])
        p = dp.BPoly.from_derivatives(xi, yi)
        assert p.degree == 3
        np.testing.assert_allclose(p(0.0), 0.0, atol=1e-10)
        np.testing.assert_allclose(p(1.0), 1.0, atol=1e-10)

    def test_bpoly_from_derivatives_tuple(self):
        """Test BPoly.from_derivatives with tuples."""
        import deriv_poly as dp
        xi = (0.0, 1.0)
        yi = ((0.0, 1.0), (1.0, -1.0))
        p = dp.BPoly.from_derivatives(xi, yi)
        assert p.degree == 3

    def test_cpoly_construction_from_numpy(self):
        """Test CPoly construction with numpy arrays."""
        import deriv_poly as dp
        coeffs = np.array([[2.0]])
        breaks = np.array([0.0, 1.0])
        p = dp.CPoly(coeffs, breaks)
        assert p.degree == 0
        np.testing.assert_allclose(p(0.5), 2.0, atol=1e-10)

    def test_cpoly_construction_from_tuple(self):
        """Test CPoly construction with tuples."""
        import deriv_poly as dp
        coeffs = ((2.0,),)
        breaks = (0.0, 1.0)
        p = dp.CPoly(coeffs, breaks)
        assert p.degree == 0

    def test_legpoly_construction_from_numpy(self):
        """Test LegPoly construction with numpy arrays."""
        import deriv_poly as dp
        coeffs = np.array([[3.0]])
        breaks = np.array([0.0, 1.0])
        p = dp.LegPoly(coeffs, breaks)
        assert p.degree == 0
        np.testing.assert_allclose(p(0.5), 3.0, atol=1e-10)

    def test_legpoly_construction_from_tuple(self):
        """Test LegPoly construction with tuples."""
        import deriv_poly as dp
        coeffs = ((3.0,),)
        breaks = (0.0, 1.0)
        p = dp.LegPoly(coeffs, breaks)
        assert p.degree == 0

    def test_bspoly_construction_from_numpy(self):
        """Test BsPoly construction with numpy arrays."""
        import deriv_poly as dp
        coeffs = np.array([[4.0]])
        breaks = np.array([0.0, 1.0])
        p = dp.BsPoly(coeffs, breaks)
        assert p.degree == 0
        np.testing.assert_allclose(p(0.5), 4.0, atol=1e-10)

    def test_bspoly_construction_from_tuple(self):
        """Test BsPoly construction with tuples."""
        import deriv_poly as dp
        coeffs = ((4.0,),)
        breaks = (0.0, 1.0)
        p = dp.BsPoly(coeffs, breaks)
        assert p.degree == 0

    def test_lagpoly_from_derivatives_numpy(self):
        """Test LagPoly.from_derivatives with numpy arrays."""
        import deriv_poly as dp
        xi = np.array([0.0, 1.0])
        yi = np.array([[0.0], [1.0]])
        p = dp.LagPoly.from_derivatives(xi, yi)
        np.testing.assert_allclose(p(0.0), 0.0, atol=1e-10)
        np.testing.assert_allclose(p(1.0), 1.0, atol=1e-10)

    def test_lagpoly_from_derivatives_tuple(self):
        """Test LagPoly.from_derivatives with tuples."""
        import deriv_poly as dp
        xi = (0.0, 1.0)
        yi = ((0.0,), (1.0,))
        p = dp.LagPoly.from_derivatives(xi, yi)
        np.testing.assert_allclose(p(0.0), 0.0, atol=1e-10)

    def test_hpoly_construction_from_numpy(self):
        """Test HPoly construction with numpy arrays."""
        import deriv_poly as dp
        coeffs = np.array([[2.0]])
        breaks = np.array([0.0, 1.0])
        p = dp.HPoly(coeffs, breaks)
        assert p.degree == 0
        np.testing.assert_allclose(p(0.5), 2.0, atol=1e-10)

    def test_hpoly_construction_from_tuple(self):
        """Test HPoly construction with tuples."""
        import deriv_poly as dp
        coeffs = ((2.0,),)
        breaks = (0.0, 1.0)
        p = dp.HPoly(coeffs, breaks)
        assert p.degree == 0

    def test_hpoly_from_derivatives_numpy(self):
        """Test HPoly.from_derivatives with numpy arrays."""
        import deriv_poly as dp
        xi = np.array([0.0, 1.0])
        yi = np.array([[0.0, 1.0], [1.0, -1.0]])
        p = dp.HPoly.from_derivatives(xi, yi)
        assert p.degree == 3
        np.testing.assert_allclose(p(0.0), 0.0, atol=1e-10)
        np.testing.assert_allclose(p(1.0), 1.0, atol=1e-10)

    def test_orders_parameter_numpy(self):
        """Test orders parameter as numpy array."""
        import deriv_poly as dp
        xi = np.array([0.0, 1.0])
        yi = np.array([[0.0, 1.0], [1.0, -1.0]])
        orders = np.array([1, 1])  # Only use 0th and 1st derivatives
        p = dp.BPoly.from_derivatives(xi, yi, orders)
        assert p.degree == 3

    def test_orders_parameter_tuple(self):
        """Test orders parameter as tuple."""
        import deriv_poly as dp
        xi = [0.0, 1.0]
        yi = [[0.0, 1.0], [1.0, -1.0]]
        orders = (1, 1)
        p = dp.BPoly.from_derivatives(xi, yi, orders)
        assert p.degree == 3

    def test_float32_numpy_array(self):
        """Test construction with float32 numpy arrays."""
        import deriv_poly as dp
        coeffs = np.array([[0.0], [1.0]], dtype=np.float32)
        breaks = np.array([0.0, 1.0], dtype=np.float32)
        p = dp.BPoly(coeffs, breaks)
        assert p.degree == 1
        np.testing.assert_allclose(p(0.5), 0.5, atol=1e-6)

    def test_extend_with_numpy(self):
        """Test extend method with numpy arrays."""
        import deriv_poly as dp
        p = dp.BPoly([[0], [1]], [0, 1])
        new_coeffs = np.array([[1.0], [2.0]])
        new_breaks = np.array([1.0, 2.0])
        p2 = p.extend(new_coeffs, new_breaks)
        assert p2.num_intervals == 2

    def test_extend_with_tuple(self):
        """Test extend method with tuples."""
        import deriv_poly as dp
        p = dp.BPoly([[0], [1]], [0, 1])
        new_coeffs = ((1.0,), (2.0,))
        new_breaks = (1.0, 2.0)
        p2 = p.extend(new_coeffs, new_breaks)
        assert p2.num_intervals == 2


class TestZeroCopy:
    """Tests for zero-copy numpy array construction and c_array property."""

    def test_numpy_zerocopy_construction(self):
        """Test that numpy arrays use zero-copy path."""
        import deriv_poly as dp
        coeffs = np.array([[0.0], [1.0]], dtype=np.float64, order='C')
        p = dp.BPoly(coeffs, [0.0, 1.0])
        assert p.degree == 1
        np.testing.assert_allclose(p(0.5), 0.5, atol=1e-10)

    def test_c_array_returns_view(self):
        """Test that c_array property returns a numpy array view."""
        import deriv_poly as dp
        coeffs = np.array([[1.0], [2.0]], dtype=np.float64)
        p = dp.BPoly(coeffs, [0, 1])
        c_arr = p.c_array
        assert isinstance(c_arr, np.ndarray)
        assert c_arr.shape == (2, 1)
        np.testing.assert_array_equal(c_arr, coeffs)

    def test_list_construction_backward_compat(self):
        """Test that list construction still works (backward compatibility)."""
        import deriv_poly as dp
        p = dp.BPoly([[0], [1]], [0, 1])
        assert p.degree == 1
        np.testing.assert_allclose(p(0.5), 0.5, atol=1e-10)

    def test_f_order_array_falls_back(self):
        """Test that Fortran-order arrays fall back to copy path."""
        import deriv_poly as dp
        coeffs = np.array([[0, 1], [1, 2]], dtype=np.float64, order='F')
        p = dp.BPoly(coeffs, [0, 0.5, 1])
        assert p.degree == 1
        # Should work correctly via copy path
        np.testing.assert_allclose(p(0.25), 0.5, atol=1e-10)

    def test_c_array_cpoly(self):
        """Test c_array property on CPoly."""
        import deriv_poly as dp
        coeffs = np.array([[2.0]], dtype=np.float64)
        p = dp.CPoly(coeffs, [0, 1])
        c_arr = p.c_array
        assert c_arr.shape == (1, 1)
        np.testing.assert_array_equal(c_arr, coeffs)

    def test_c_array_legpoly(self):
        """Test c_array property on LegPoly."""
        import deriv_poly as dp
        coeffs = np.array([[3.0]], dtype=np.float64)
        p = dp.LegPoly(coeffs, [0, 1])
        c_arr = p.c_array
        assert c_arr.shape == (1, 1)
        np.testing.assert_array_equal(c_arr, coeffs)

    def test_c_array_bspoly(self):
        """Test c_array property on BsPoly."""
        import deriv_poly as dp
        coeffs = np.array([[4.0]], dtype=np.float64)
        p = dp.BsPoly(coeffs, [0, 1])
        c_arr = p.c_array
        assert c_arr.shape == (1, 1)
        np.testing.assert_array_equal(c_arr, coeffs)

    def test_c_array_hpoly(self):
        """Test c_array property on HPoly."""
        import deriv_poly as dp
        coeffs = np.array([[5.0]], dtype=np.float64)
        p = dp.HPoly(coeffs, [0, 1])
        c_arr = p.c_array
        assert c_arr.shape == (1, 1)
        np.testing.assert_array_equal(c_arr, coeffs)

    def test_c_array_shape_multi_interval(self):
        """Test c_array shape with multiple intervals."""
        import deriv_poly as dp
        # 3 coefficients per interval, 2 intervals
        coeffs = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float64)
        p = dp.BPoly(coeffs, [0, 0.5, 1])
        c_arr = p.c_array
        assert c_arr.shape == (3, 2)
        np.testing.assert_array_equal(c_arr, coeffs)


class TestMixedPolyList:
    """Tests for MixedPolyList with HPoly."""

    def test_mixed_list_with_hpoly(self):
        """Test MixedPolyList containing HPoly."""
        import deriv_poly as dp
        bp = dp.BPoly([[1]], [0, 1])
        cp = dp.CPoly([[2]], [0, 1])
        hp = dp.HPoly([[3]], [0, 1])
        ml = dp.MixedPolyList([bp, cp, hp])

        assert len(ml) == 3
        y = ml(0.5)
        np.testing.assert_allclose(y, [1.0, 2.0, 3.0], atol=1e-10)

    def test_mixed_list_append_hpoly(self):
        """Test appending HPoly to MixedPolyList."""
        import deriv_poly as dp
        ml = dp.MixedPolyList()
        hp = dp.HPoly([[5]], [0, 1])
        ml.append(hp)

        assert len(ml) == 1
        y = ml(0.5)
        np.testing.assert_allclose(y, [5.0], atol=1e-10)

    def test_mixed_list_hpoly_integration(self):
        """Test MixedPolyList integration with HPoly."""
        import deriv_poly as dp
        bp = dp.BPoly([[1]], [0, 1])
        hp = dp.HPoly([[2]], [0, 1])
        ml = dp.MixedPolyList([bp, hp])

        result = ml.integrate(0, 1)
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-10)

    def test_mixed_list_hpoly_both_kinds(self):
        """Test MixedPolyList with both Hermite variants."""
        import deriv_poly as dp
        hp_phys = dp.HPoly([[1]], [0, 1], kind=dp.HermiteKind.Physicist)
        hp_prob = dp.HPoly([[1]], [0, 1], kind=dp.HermiteKind.Probabilist)
        ml = dp.MixedPolyList([hp_phys, hp_prob])

        y = ml(0.5)
        np.testing.assert_allclose(y, [1.0, 1.0], atol=1e-10)


class TestScipyComparison:
    """Compare results against scipy for accuracy verification."""

    @pytest.mark.skipif(True, reason="Requires scipy")
    def test_bpoly_vs_scipy(self):
        """Compare BPoly against scipy.interpolate.BPoly."""
        import deriv_poly as dp
        from scipy.interpolate import BPoly as ScipyBPoly
        import numpy as np

        xi = [0, 1, 2]
        yi = [[0, 1], [1, -1], [0.5, 0.5]]

        bp_cpp = dp.BPoly.from_derivatives(xi, yi)
        bp_scipy = ScipyBPoly.from_derivatives(xi, yi)

        x = np.linspace(0, 2, 100)
        y_cpp = bp_cpp(x)
        y_scipy = bp_scipy(x)

        np.testing.assert_allclose(y_cpp, y_scipy, rtol=1e-10)


class TestPerformance:
    """Performance sanity checks."""

    def test_list_no_python_loop(self):
        """
        Verify that list evaluation is vectorized (no Python for-loop).
        This test creates many polynomials and evaluates at many points.
        If it's too slow, Python for-loops are likely involved.
        """
        import deriv_poly as dp
        import time

        # Create 100 polynomials
        polys = []
        for i in range(100):
            p = dp.BPoly([[float(i)]], [0, 1])
            polys.append(p)
        pl = dp.BPolyList(polys)

        # Evaluate at 10000 points
        x = np.linspace(0, 1, 10000)

        start = time.time()
        y = pl(x)
        elapsed = time.time() - start

        # Should complete in under 1 second on any modern machine
        # If Python for-loops were involved, it would be much slower
        assert elapsed < 1.0
        assert y.shape == (100, 10000)


class TestCPolyComprehensive:
    """Comprehensive tests for CPoly class."""

    def test_from_derivatives(self):
        """Test CPoly.from_derivatives construction and endpoint values."""
        import deriv_poly as dp
        # f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
        p = dp.CPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        assert abs(p(0.0) - 0.0) < 1e-10
        assert abs(p(1.0) - 1.0) < 1e-10
        assert p.degree == 3

    def test_evaluation_array(self):
        """Test CPoly array evaluation with np.linspace."""
        import deriv_poly as dp
        p = dp.CPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        x = np.linspace(0, 1, 50)
        y = p(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == (50,)
        # Check endpoints
        np.testing.assert_allclose(y[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(y[-1], 1.0, atol=1e-10)

    def test_derivative_nu(self):
        """Test CPoly derivative evaluation via nu parameter."""
        import deriv_poly as dp
        p = dp.CPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        # f'(0) should be 1.0
        np.testing.assert_allclose(p(0.0, nu=1), 1.0, atol=1e-10)
        # f'(1) should be -1.0
        np.testing.assert_allclose(p(1.0, nu=1), -1.0, atol=1e-10)

    def test_derivative_method(self):
        """Test CPoly .derivative() returns a polynomial with correct values."""
        import deriv_poly as dp
        p = dp.CPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        dp_poly = p.derivative()
        np.testing.assert_allclose(dp_poly(0.0), 1.0, atol=1e-10)
        np.testing.assert_allclose(dp_poly(1.0), -1.0, atol=1e-10)

    def test_integration(self):
        """Test CPoly definite integration."""
        import deriv_poly as dp
        # Constant polynomial f(x) = 3 on [0, 2]
        p = dp.CPoly([[3]], [0, 2])
        result = p.integrate(0, 2)
        np.testing.assert_allclose(result, 6.0, atol=1e-10)

    def test_roots(self):
        """Test CPoly root finding."""
        import deriv_poly as dp
        # f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1 -- this passes through zero at x=0
        p = dp.CPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        roots = p.roots()
        assert len(roots) >= 1
        # x=0 should be a root
        assert any(abs(r) < 1e-10 for r in roots)

    def test_extrapolation_modes(self):
        """Test CPoly NoExtrapolate returns NaN outside domain."""
        import deriv_poly as dp
        p = dp.CPoly([[1]], [0, 1], extrapolate=dp.ExtrapolateMode.NoExtrapolate)
        # Inside domain should be fine
        np.testing.assert_allclose(p(0.5), 1.0, atol=1e-10)
        # Outside domain should be NaN
        assert np.isnan(p(-0.5))
        assert np.isnan(p(1.5))

    def test_to_power_basis(self):
        """Test CPoly from_power_basis then to_power_basis roundtrip."""
        import deriv_poly as dp
        # Power basis coefficients: [c0, c1, c2] for each interval
        # f(x) = 1 + 2*(x - a) + 3*(x - a)^2 on [0, 1]
        power_coeffs = [[1], [2], [3]]
        p = dp.CPoly.from_power_basis(power_coeffs, [0, 1])
        recovered = p.to_power_basis()
        np.testing.assert_allclose(recovered[0][0], 1.0, atol=1e-10)
        np.testing.assert_allclose(recovered[1][0], 2.0, atol=1e-10)
        np.testing.assert_allclose(recovered[2][0], 3.0, atol=1e-10)


class TestLegPolyComprehensive:
    """Comprehensive tests for LegPoly class."""

    def test_from_derivatives(self):
        """Test LegPoly.from_derivatives construction and endpoint values."""
        import deriv_poly as dp
        p = dp.LegPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        assert abs(p(0.0) - 0.0) < 1e-10
        assert abs(p(1.0) - 1.0) < 1e-10
        assert p.degree == 3

    def test_evaluation_array(self):
        """Test LegPoly array evaluation with np.linspace."""
        import deriv_poly as dp
        p = dp.LegPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        x = np.linspace(0, 1, 50)
        y = p(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == (50,)
        np.testing.assert_allclose(y[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(y[-1], 1.0, atol=1e-10)

    def test_derivative_nu(self):
        """Test LegPoly derivative evaluation via nu parameter."""
        import deriv_poly as dp
        p = dp.LegPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        np.testing.assert_allclose(p(0.0, nu=1), 1.0, atol=1e-10)
        np.testing.assert_allclose(p(1.0, nu=1), -1.0, atol=1e-10)

    def test_derivative_method(self):
        """Test LegPoly .derivative() returns a polynomial with correct values."""
        import deriv_poly as dp
        p = dp.LegPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        dp_poly = p.derivative()
        np.testing.assert_allclose(dp_poly(0.0), 1.0, atol=1e-10)
        np.testing.assert_allclose(dp_poly(1.0), -1.0, atol=1e-10)

    def test_integration(self):
        """Test LegPoly definite integration."""
        import deriv_poly as dp
        p = dp.LegPoly([[3]], [0, 2])
        result = p.integrate(0, 2)
        np.testing.assert_allclose(result, 6.0, atol=1e-10)

    def test_roots(self):
        """Test LegPoly root finding."""
        import deriv_poly as dp
        p = dp.LegPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        roots = p.roots()
        assert len(roots) >= 1
        assert any(abs(r) < 1e-10 for r in roots)

    def test_extrapolation_modes(self):
        """Test LegPoly NoExtrapolate returns NaN outside domain."""
        import deriv_poly as dp
        p = dp.LegPoly([[1]], [0, 1], extrapolate=dp.ExtrapolateMode.NoExtrapolate)
        np.testing.assert_allclose(p(0.5), 1.0, atol=1e-10)
        assert np.isnan(p(-0.5))
        assert np.isnan(p(1.5))

    def test_to_power_basis(self):
        """Test LegPoly from_power_basis then to_power_basis roundtrip."""
        import deriv_poly as dp
        power_coeffs = [[1], [2], [3]]
        p = dp.LegPoly.from_power_basis(power_coeffs, [0, 1])
        recovered = p.to_power_basis()
        np.testing.assert_allclose(recovered[0][0], 1.0, atol=1e-10)
        np.testing.assert_allclose(recovered[1][0], 2.0, atol=1e-10)
        np.testing.assert_allclose(recovered[2][0], 3.0, atol=1e-10)


class TestBsPolyComprehensive:
    """Comprehensive tests for BsPoly class."""

    def test_from_derivatives(self):
        """Test BsPoly.from_derivatives construction and endpoint values."""
        import deriv_poly as dp
        p = dp.BsPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        assert abs(p(0.0) - 0.0) < 1e-10
        assert abs(p(1.0) - 1.0) < 1e-10
        assert p.degree == 3

    def test_evaluation_array(self):
        """Test BsPoly array evaluation with np.linspace."""
        import deriv_poly as dp
        p = dp.BsPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        x = np.linspace(0, 1, 50)
        y = p(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == (50,)
        np.testing.assert_allclose(y[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(y[-1], 1.0, atol=1e-10)

    def test_derivative_nu(self):
        """Test BsPoly derivative evaluation via nu parameter."""
        import deriv_poly as dp
        p = dp.BsPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        np.testing.assert_allclose(p(0.0, nu=1), 1.0, atol=1e-10)
        np.testing.assert_allclose(p(1.0, nu=1), -1.0, atol=1e-10)

    def test_derivative_method(self):
        """Test BsPoly .derivative() returns a polynomial with correct values."""
        import deriv_poly as dp
        p = dp.BsPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        dp_poly = p.derivative()
        np.testing.assert_allclose(dp_poly(0.0), 1.0, atol=1e-10)
        np.testing.assert_allclose(dp_poly(1.0), -1.0, atol=1e-10)

    def test_integration(self):
        """Test BsPoly definite integration."""
        import deriv_poly as dp
        p = dp.BsPoly([[3]], [0, 2])
        result = p.integrate(0, 2)
        np.testing.assert_allclose(result, 6.0, atol=1e-10)

    def test_roots(self):
        """Test BsPoly root finding."""
        import deriv_poly as dp
        p = dp.BsPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        roots = p.roots()
        assert len(roots) >= 1
        assert any(abs(r) < 1e-10 for r in roots)

    def test_extrapolation_modes(self):
        """Test BsPoly NoExtrapolate returns NaN outside domain."""
        import deriv_poly as dp
        p = dp.BsPoly([[1]], [0, 1], extrapolate=dp.ExtrapolateMode.NoExtrapolate)
        np.testing.assert_allclose(p(0.5), 1.0, atol=1e-10)
        assert np.isnan(p(-0.5))
        assert np.isnan(p(1.5))

    def test_to_power_basis(self):
        """Test BsPoly from_power_basis then to_power_basis roundtrip."""
        import deriv_poly as dp
        power_coeffs = [[1], [2], [3]]
        p = dp.BsPoly.from_power_basis(power_coeffs, [0, 1])
        recovered = p.to_power_basis()
        np.testing.assert_allclose(recovered[0][0], 1.0, atol=1e-10)
        np.testing.assert_allclose(recovered[1][0], 2.0, atol=1e-10)
        np.testing.assert_allclose(recovered[2][0], 3.0, atol=1e-10)


class TestLagPolyComprehensive:
    """Comprehensive tests for LagPoly class."""

    def test_from_derivatives(self):
        """Test LagPoly.from_derivatives construction and endpoint values."""
        import deriv_poly as dp
        # f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
        p = dp.LagPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        np.testing.assert_allclose(p(0.0), 0.0, atol=1e-10)
        np.testing.assert_allclose(p(1.0), 1.0, atol=1e-10)

    def test_evaluation_array(self):
        """Test LagPoly array evaluation with np.linspace."""
        import deriv_poly as dp
        p = dp.LagPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        x = np.linspace(0, 1, 50)
        y = p(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == (50,)
        np.testing.assert_allclose(y[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(y[-1], 1.0, atol=1e-10)

    def test_derivative_nu(self):
        """Test LagPoly derivative evaluation via nu parameter."""
        import deriv_poly as dp
        p = dp.LagPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        np.testing.assert_allclose(p(0.0, nu=1), 1.0, atol=1e-10)
        np.testing.assert_allclose(p(1.0, nu=1), -1.0, atol=1e-10)

    def test_derivative_method(self):
        """Test LagPoly .derivative() returns a polynomial with correct values."""
        import deriv_poly as dp
        p = dp.LagPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        dp_poly = p.derivative()
        np.testing.assert_allclose(dp_poly(0.0), 1.0, atol=1e-10)
        np.testing.assert_allclose(dp_poly(1.0), -1.0, atol=1e-10)

    def test_integration(self):
        """Test LagPoly definite integration."""
        import deriv_poly as dp
        # f(0)=3, f(1)=3 -> constant 3, integral from 0 to 1 = 3
        p = dp.LagPoly.from_derivatives([0, 1], [[3], [3]])
        result = p.integrate(0, 1)
        np.testing.assert_allclose(result, 3.0, atol=1e-10)

    def test_roots(self):
        """Test LagPoly root finding."""
        import deriv_poly as dp
        # f(0)=-1, f(1)=1 -> linear, root at x=0.5
        p = dp.LagPoly.from_derivatives([0, 1], [[-1], [1]])
        roots = p.roots()
        assert len(roots) >= 1
        assert any(abs(r - 0.5) < 1e-10 for r in roots)

    def test_nodes_values_weights(self):
        """Test LagPoly accessor properties: nodes, values, weights."""
        import deriv_poly as dp
        p = dp.LagPoly.from_derivatives([0, 1], [[0], [1]])
        nodes = p.nodes
        values = p.values
        weights = p.weights
        assert isinstance(nodes, list)
        assert isinstance(values, list)
        assert isinstance(weights, list)
        assert len(nodes) >= 1
        assert len(values) >= 1
        assert len(weights) >= 1

    def test_to_power_basis(self):
        """Test LagPoly to_power_basis."""
        import deriv_poly as dp
        # Linear: f(0)=0, f(1)=1 -> f(x) = x, power basis [0, 1]
        p = dp.LagPoly.from_derivatives([0, 1], [[0], [1]])
        power = p.to_power_basis()
        assert isinstance(power, list)
        # Should have 2 rows (constant and linear terms)
        assert len(power) >= 1


class TestCPolyList:
    """Tests for CPolyList class."""

    def test_construction_and_len(self):
        """Test CPolyList construction and length."""
        import deriv_poly as dp
        p1 = dp.CPoly([[1]], [0, 1])
        p2 = dp.CPoly([[2]], [0, 1])
        p3 = dp.CPoly([[3]], [0, 1])
        pl = dp.CPolyList([p1, p2, p3])
        assert len(pl) == 3

    def test_batch_scalar_eval(self):
        """Test CPolyList scalar evaluation returns (N,) array."""
        import deriv_poly as dp
        p1 = dp.CPoly([[1]], [0, 1])
        p2 = dp.CPoly([[2]], [0, 1])
        pl = dp.CPolyList([p1, p2])
        y = pl(0.5)
        assert isinstance(y, np.ndarray)
        assert y.shape == (2,)
        np.testing.assert_allclose(y, [1.0, 2.0], atol=1e-10)

    def test_batch_array_eval(self):
        """Test CPolyList array evaluation returns (N, M) array."""
        import deriv_poly as dp
        p1 = dp.CPoly([[1]], [0, 1])
        p2 = dp.CPoly([[2]], [0, 1])
        pl = dp.CPolyList([p1, p2])
        x = np.array([0.1, 0.5, 0.9])
        y = pl(x)
        assert y.shape == (2, 3)
        np.testing.assert_allclose(y[0], [1.0, 1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(y[1], [2.0, 2.0, 2.0], atol=1e-10)

    def test_batch_integration(self):
        """Test CPolyList batch integration."""
        import deriv_poly as dp
        p1 = dp.CPoly([[1]], [0, 1])  # integral = 1
        p2 = dp.CPoly([[2]], [0, 1])  # integral = 2
        pl = dp.CPolyList([p1, p2])
        result = pl.integrate(0, 1)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-10)

    def test_batch_derivative(self):
        """Test CPolyList derivative returns new list."""
        import deriv_poly as dp
        p1 = dp.CPoly.from_derivatives([0, 1], [[0, 1], [1, 0]])
        p2 = dp.CPoly.from_derivatives([0, 1], [[0, 2], [1, 0]])
        pl = dp.CPolyList([p1, p2])
        dpl = pl.derivative()
        assert len(dpl) == 2
        # f'(0) should be 1.0 for p1
        np.testing.assert_allclose(dpl[0](0.0), 1.0, atol=1e-10)

    def test_setitem(self):
        """Test CPolyList __setitem__."""
        import deriv_poly as dp
        p1 = dp.CPoly([[1]], [0, 1])
        p2 = dp.CPoly([[2]], [0, 1])
        pl = dp.CPolyList([p1, p2])
        p_new = dp.CPoly([[7]], [0, 1])
        pl[1] = p_new
        y = pl(0.5)
        np.testing.assert_allclose(y, [1.0, 7.0], atol=1e-10)


class TestLegPolyList:
    """Tests for LegPolyList class."""

    def test_construction_and_len(self):
        """Test LegPolyList construction and length."""
        import deriv_poly as dp
        p1 = dp.LegPoly([[1]], [0, 1])
        p2 = dp.LegPoly([[2]], [0, 1])
        p3 = dp.LegPoly([[3]], [0, 1])
        pl = dp.LegPolyList([p1, p2, p3])
        assert len(pl) == 3

    def test_batch_scalar_eval(self):
        """Test LegPolyList scalar evaluation returns (N,) array."""
        import deriv_poly as dp
        p1 = dp.LegPoly([[1]], [0, 1])
        p2 = dp.LegPoly([[2]], [0, 1])
        pl = dp.LegPolyList([p1, p2])
        y = pl(0.5)
        assert isinstance(y, np.ndarray)
        assert y.shape == (2,)
        np.testing.assert_allclose(y, [1.0, 2.0], atol=1e-10)

    def test_batch_array_eval(self):
        """Test LegPolyList array evaluation returns (N, M) array."""
        import deriv_poly as dp
        p1 = dp.LegPoly([[1]], [0, 1])
        p2 = dp.LegPoly([[2]], [0, 1])
        pl = dp.LegPolyList([p1, p2])
        x = np.array([0.1, 0.5, 0.9])
        y = pl(x)
        assert y.shape == (2, 3)
        np.testing.assert_allclose(y[0], [1.0, 1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(y[1], [2.0, 2.0, 2.0], atol=1e-10)

    def test_batch_integration(self):
        """Test LegPolyList batch integration."""
        import deriv_poly as dp
        p1 = dp.LegPoly([[1]], [0, 1])
        p2 = dp.LegPoly([[2]], [0, 1])
        pl = dp.LegPolyList([p1, p2])
        result = pl.integrate(0, 1)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-10)

    def test_batch_derivative(self):
        """Test LegPolyList derivative returns new list."""
        import deriv_poly as dp
        p1 = dp.LegPoly.from_derivatives([0, 1], [[0, 1], [1, 0]])
        p2 = dp.LegPoly.from_derivatives([0, 1], [[0, 2], [1, 0]])
        pl = dp.LegPolyList([p1, p2])
        dpl = pl.derivative()
        assert len(dpl) == 2
        np.testing.assert_allclose(dpl[0](0.0), 1.0, atol=1e-10)

    def test_setitem(self):
        """Test LegPolyList __setitem__."""
        import deriv_poly as dp
        p1 = dp.LegPoly([[1]], [0, 1])
        p2 = dp.LegPoly([[2]], [0, 1])
        pl = dp.LegPolyList([p1, p2])
        p_new = dp.LegPoly([[7]], [0, 1])
        pl[0] = p_new
        y = pl(0.5)
        np.testing.assert_allclose(y, [7.0, 2.0], atol=1e-10)


class TestBsPolyList:
    """Tests for BsPolyList class."""

    def test_construction_and_len(self):
        """Test BsPolyList construction and length."""
        import deriv_poly as dp
        p1 = dp.BsPoly([[1]], [0, 1])
        p2 = dp.BsPoly([[2]], [0, 1])
        p3 = dp.BsPoly([[3]], [0, 1])
        pl = dp.BsPolyList([p1, p2, p3])
        assert len(pl) == 3

    def test_batch_scalar_eval(self):
        """Test BsPolyList scalar evaluation returns (N,) array."""
        import deriv_poly as dp
        p1 = dp.BsPoly([[1]], [0, 1])
        p2 = dp.BsPoly([[2]], [0, 1])
        pl = dp.BsPolyList([p1, p2])
        y = pl(0.5)
        assert isinstance(y, np.ndarray)
        assert y.shape == (2,)
        np.testing.assert_allclose(y, [1.0, 2.0], atol=1e-10)

    def test_batch_array_eval(self):
        """Test BsPolyList array evaluation returns (N, M) array."""
        import deriv_poly as dp
        p1 = dp.BsPoly([[1]], [0, 1])
        p2 = dp.BsPoly([[2]], [0, 1])
        pl = dp.BsPolyList([p1, p2])
        x = np.array([0.1, 0.5, 0.9])
        y = pl(x)
        assert y.shape == (2, 3)
        np.testing.assert_allclose(y[0], [1.0, 1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(y[1], [2.0, 2.0, 2.0], atol=1e-10)

    def test_batch_integration(self):
        """Test BsPolyList batch integration."""
        import deriv_poly as dp
        p1 = dp.BsPoly([[1]], [0, 1])
        p2 = dp.BsPoly([[2]], [0, 1])
        pl = dp.BsPolyList([p1, p2])
        result = pl.integrate(0, 1)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-10)

    def test_batch_derivative(self):
        """Test BsPolyList derivative returns new list."""
        import deriv_poly as dp
        p1 = dp.BsPoly.from_derivatives([0, 1], [[0, 1], [1, 0]])
        p2 = dp.BsPoly.from_derivatives([0, 1], [[0, 2], [1, 0]])
        pl = dp.BsPolyList([p1, p2])
        dpl = pl.derivative()
        assert len(dpl) == 2
        np.testing.assert_allclose(dpl[0](0.0), 1.0, atol=1e-10)

    def test_setitem(self):
        """Test BsPolyList __setitem__."""
        import deriv_poly as dp
        p1 = dp.BsPoly([[1]], [0, 1])
        p2 = dp.BsPoly([[2]], [0, 1])
        pl = dp.BsPolyList([p1, p2])
        p_new = dp.BsPoly([[7]], [0, 1])
        pl[1] = p_new
        y = pl(0.5)
        np.testing.assert_allclose(y, [1.0, 7.0], atol=1e-10)


class TestLagPolyList:
    """Tests for LagPolyList class."""

    def test_construction_and_len(self):
        """Test LagPolyList construction and length."""
        import deriv_poly as dp
        p1 = dp.LagPoly.from_derivatives([0, 1], [[1], [1]])
        p2 = dp.LagPoly.from_derivatives([0, 1], [[2], [2]])
        p3 = dp.LagPoly.from_derivatives([0, 1], [[3], [3]])
        pl = dp.LagPolyList([p1, p2, p3])
        assert len(pl) == 3

    def test_batch_scalar_eval(self):
        """Test LagPolyList scalar evaluation returns (N,) array."""
        import deriv_poly as dp
        p1 = dp.LagPoly.from_derivatives([0, 1], [[1], [1]])
        p2 = dp.LagPoly.from_derivatives([0, 1], [[2], [2]])
        pl = dp.LagPolyList([p1, p2])
        y = pl(0.5)
        assert isinstance(y, np.ndarray)
        assert y.shape == (2,)
        np.testing.assert_allclose(y, [1.0, 2.0], atol=1e-10)

    def test_batch_array_eval(self):
        """Test LagPolyList array evaluation returns (N, M) array."""
        import deriv_poly as dp
        p1 = dp.LagPoly.from_derivatives([0, 1], [[1], [1]])
        p2 = dp.LagPoly.from_derivatives([0, 1], [[2], [2]])
        pl = dp.LagPolyList([p1, p2])
        x = np.array([0.1, 0.5, 0.9])
        y = pl(x)
        assert y.shape == (2, 3)
        np.testing.assert_allclose(y[0], [1.0, 1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(y[1], [2.0, 2.0, 2.0], atol=1e-10)

    def test_batch_integration(self):
        """Test LagPolyList batch integration."""
        import deriv_poly as dp
        p1 = dp.LagPoly.from_derivatives([0, 1], [[1], [1]])
        p2 = dp.LagPoly.from_derivatives([0, 1], [[2], [2]])
        pl = dp.LagPolyList([p1, p2])
        result = pl.integrate(0, 1)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-10)

    def test_batch_derivative(self):
        """Test LagPolyList derivative returns new list."""
        import deriv_poly as dp
        p1 = dp.LagPoly.from_derivatives([0, 1], [[0, 1], [1, 0]])
        p2 = dp.LagPoly.from_derivatives([0, 1], [[0, 2], [1, 0]])
        pl = dp.LagPolyList([p1, p2])
        dpl = pl.derivative()
        assert len(dpl) == 2
        np.testing.assert_allclose(dpl[0](0.0), 1.0, atol=1e-10)

    def test_setitem(self):
        """Test LagPolyList __setitem__."""
        import deriv_poly as dp
        p1 = dp.LagPoly.from_derivatives([0, 1], [[1], [1]])
        p2 = dp.LagPoly.from_derivatives([0, 1], [[2], [2]])
        pl = dp.LagPolyList([p1, p2])
        p_new = dp.LagPoly.from_derivatives([0, 1], [[7], [7]])
        pl[0] = p_new
        y = pl(0.5)
        np.testing.assert_allclose(y, [7.0, 2.0], atol=1e-10)


class TestMixedPolyListExpanded:
    """Expanded tests for MixedPolyList."""

    def test_array_evaluation(self):
        """Test MixedPolyList multi-point evaluation with numpy array."""
        import deriv_poly as dp
        bp = dp.BPoly([[1]], [0, 1])
        cp = dp.CPoly([[2]], [0, 1])
        hp = dp.HPoly([[3]], [0, 1])
        ml = dp.MixedPolyList([bp, cp, hp])
        x = np.array([0.2, 0.5, 0.8])
        y = ml(x)
        assert y.shape == (3, 3)
        np.testing.assert_allclose(y[0], [1.0, 1.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(y[1], [2.0, 2.0, 2.0], atol=1e-10)
        np.testing.assert_allclose(y[2], [3.0, 3.0, 3.0], atol=1e-10)

    def test_nu_parameter(self):
        """Test MixedPolyList derivative evaluation with nu=1."""
        import deriv_poly as dp
        bp = dp.BPoly.from_derivatives([0, 1], [[0, 1], [1, 0]])
        cp = dp.CPoly.from_derivatives([0, 1], [[0, 2], [1, 0]])
        ml = dp.MixedPolyList([bp, cp])
        y = ml(0.0, nu=1)
        assert y.shape == (2,)
        np.testing.assert_allclose(y[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(y[1], 2.0, atol=1e-10)

    def test_getitem_returns_correct_type(self):
        """Test MixedPolyList __getitem__ returns correct polynomial types."""
        import deriv_poly as dp
        bp = dp.BPoly([[1]], [0, 1])
        cp = dp.CPoly([[2]], [0, 1])
        lp = dp.LegPoly([[3]], [0, 1])
        hp = dp.HPoly([[4]], [0, 1])
        bsp = dp.BsPoly([[5]], [0, 1])
        lagp = dp.LagPoly.from_derivatives([0, 1], [[6], [6]])
        ml = dp.MixedPolyList([bp, cp, lp, hp, bsp, lagp])

        assert isinstance(ml[0], dp.BPoly)
        assert isinstance(ml[1], dp.CPoly)
        assert isinstance(ml[2], dp.LegPoly)
        assert isinstance(ml[3], dp.HPoly)
        assert isinstance(ml[4], dp.BsPoly)
        assert isinstance(ml[5], dp.LagPoly)

    def test_setitem(self):
        """Test MixedPolyList __setitem__ replaces polynomial."""
        import deriv_poly as dp
        bp = dp.BPoly([[1]], [0, 1])
        cp = dp.CPoly([[2]], [0, 1])
        ml = dp.MixedPolyList([bp, cp])
        # Replace first with an HPoly
        hp = dp.HPoly([[9]], [0, 1])
        ml[0] = hp
        y = ml(0.5)
        np.testing.assert_allclose(y, [9.0, 2.0], atol=1e-10)
        assert isinstance(ml[0], dp.HPoly)

    def test_extend_and_clear(self):
        """Test MixedPolyList extend then clear."""
        import deriv_poly as dp
        ml = dp.MixedPolyList()
        bp = dp.BPoly([[1]], [0, 1])
        cp = dp.CPoly([[2]], [0, 1])
        ml.extend([bp, cp])
        assert len(ml) == 2
        ml.clear()
        assert len(ml) == 0

    def test_derivative_method(self):
        """Test MixedPolyList .derivative() returns new MixedPolyList."""
        import deriv_poly as dp
        bp = dp.BPoly.from_derivatives([0, 1], [[0, 1], [1, 0]])
        cp = dp.CPoly.from_derivatives([0, 1], [[0, 2], [1, 0]])
        ml = dp.MixedPolyList([bp, cp])
        dml = ml.derivative()
        assert isinstance(dml, dp.MixedPolyList)
        assert len(dml) == 2
        # Check derivative values at x=0
        y = dml(0.0)
        np.testing.assert_allclose(y[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(y[1], 2.0, atol=1e-10)

    def test_antiderivative_method(self):
        """Test MixedPolyList .antiderivative() returns new MixedPolyList."""
        import deriv_poly as dp
        # Constant polynomials: f(x) = 1 and f(x) = 2
        bp = dp.BPoly([[1]], [0, 1])
        cp = dp.CPoly([[2]], [0, 1])
        ml = dp.MixedPolyList([bp, cp])
        aml = ml.antiderivative()
        assert isinstance(aml, dp.MixedPolyList)
        assert len(aml) == 2
        # Antiderivative of constant c is c*x (with integration constant 0 at left endpoint)
        # For BPoly with f(x)=1 on [0,1]: antiderivative at x=1 should be 1.0
        np.testing.assert_allclose(aml[0](1.0), 1.0, atol=1e-10)
        # For CPoly with f(x)=2 on [0,1]: antiderivative at x=1 should be 2.0
        np.testing.assert_allclose(aml[1](1.0), 2.0, atol=1e-10)

    def test_invalid_type_raises(self):
        """Test MixedPolyList raises error for non-polynomial objects."""
        import deriv_poly as dp
        with pytest.raises((RuntimeError, TypeError)):
            dp.MixedPolyList([42])
        with pytest.raises((RuntimeError, TypeError)):
            dp.MixedPolyList(["not a poly"])
        ml = dp.MixedPolyList()
        with pytest.raises((RuntimeError, TypeError)):
            ml.append("not a poly")


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_dtype_bpoly(self):
        """Test BPoly with int32 array either works or raises cleanly."""
        import deriv_poly as dp
        coeffs = np.array([[0], [1]], dtype=np.int32)
        try:
            p = dp.BPoly(coeffs, [0, 1])
            # If it works, it should give reasonable results
            result = p(0.5)
            assert np.isfinite(result)
        except (TypeError, RuntimeError):
            # Clean error is acceptable
            pass

    def test_mismatched_coefficients(self):
        """Test mismatched number of intervals vs breakpoints raises error."""
        import deriv_poly as dp
        # 2 intervals worth of breakpoints [0, 0.5, 1] but only 1 interval of coefficients
        # coeffs shape [2][1] means degree=1, 1 interval
        # breakpoints [0, 0.5, 1] means 2 intervals
        with pytest.raises((RuntimeError, ValueError)):
            dp.BPoly([[0, 1], [1, 2]], [0, 0.5, 1, 2])

    def test_polylist_out_of_range(self):
        """Test PolyList indexing out of range raises IndexError."""
        import deriv_poly as dp
        p1 = dp.BPoly([[1]], [0, 1])
        pl = dp.BPolyList([p1])
        with pytest.raises(IndexError):
            _ = pl[999]
        with pytest.raises(IndexError):
            pl[999] = p1

    def test_empty_breakpoints(self):
        """Test construction with empty or insufficient breakpoints raises error."""
        import deriv_poly as dp
        with pytest.raises((RuntimeError, ValueError)):
            dp.BPoly([[1]], [])
        with pytest.raises((RuntimeError, ValueError)):
            dp.BPoly([[1]], [0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
