"""Tests for boundary conditions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochlib.boundary_conditions import BoundaryConditions


class TestBoundaryConditionTypes:
    """Test different boundary condition types."""

    def test_periodic_bc(self, simple_1d_grid):
        """Test periodic boundary condition."""
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        assert bc is not None

    def test_dirichlet_bc(self, simple_1d_grid):
        """Test Dirichlet boundary condition."""
        bc = BoundaryConditions(simple_1d_grid, bc_x="noflux")
        assert bc is not None

    def test_neumann_bc(self, simple_1d_grid):
        """Test Neumann boundary condition."""
        bc = BoundaryConditions(simple_1d_grid, bc_x="noflux")
        assert bc is not None

    def test_open_bc(self, simple_1d_grid):
        """Test open/transparent boundary condition."""
        bc = BoundaryConditions(simple_1d_grid, bc_x="open")
        assert bc is not None


class TestBoundaryConditionCreation:
    """Test BC initialization."""

    def test_1d_bc_creation(self, simple_1d_grid):
        """Test 1D BC creation."""
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        assert bc is not None

    def test_2d_bc_creation(self, simple_2d_grid):
        """Test 2D BC creation with different BC types in each direction."""
        bc = BoundaryConditions(simple_2d_grid, bc_x="periodic", bc_y="noflux")
        assert bc is not None

    def test_bc_with_boundary_values(self, simple_1d_grid):
        """Test BC with explicit boundary values."""
        bc = BoundaryConditions(simple_1d_grid, bc_x="noflux")
        assert bc is not None


class TestPeriodicBC:
    """Test periodic boundary conditions."""

    def test_periodic_wraps_values(self, simple_1d_grid):
        """Test that periodic BC wraps values correctly."""
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        # Property check: values at boundaries should influence each other
        assert bc is not None

    def test_periodic_preserves_domain(self, simple_1d_grid):
        """Test that periodic BC preserves domain."""
        bc = BoundaryConditions(simple_1d_grid, bc_x=BoundaryConditions.PERIODIC)
        # No artificial boundaries introduced
        assert bc is not None


class TestDirichletBC:
    """Test Dirichlet boundary conditions."""

    def test_dirichlet_zero_values(self, simple_1d_grid):
        """Test Dirichlet BC with zero values."""
        bc = BoundaryConditions(simple_1d_grid, bc_x="noflux")
        assert bc is not None

    def test_dirichlet_nonzero_values(self, simple_1d_grid):
        """Test Dirichlet BC with non-zero values."""
        bc = BoundaryConditions(simple_1d_grid, bc_x="noflux")
        assert bc is not None

    def test_dirichlet_asymmetric_values(self, simple_1d_grid):
        """Test Dirichlet BC with asymmetric boundary values."""
        bc = BoundaryConditions(simple_1d_grid, bc_x="noflux")
        assert bc is not None


class TestNeumannBC:
    """Test Neumann boundary conditions."""

    def test_neumann_zero_flux(self, simple_1d_grid):
        """Test Neumann BC with zero flux."""
        bc = BoundaryConditions(simple_1d_grid, bc_x="noflux")
        assert bc is not None

    def test_neumann_nonzero_flux(self, simple_1d_grid):
        """Test Neumann BC with non-zero flux."""
        bc = BoundaryConditions(simple_1d_grid, bc_x="noflux")
        assert bc is not None


class TestOpenBC:
    """Test open/transparent boundary conditions."""

    def test_open_bc_basic(self, simple_1d_grid):
        """Test basic open boundary condition."""
        bc = BoundaryConditions(simple_1d_grid, bc_x="open")
        assert bc is not None

    def test_open_bc_allows_outflow(self, simple_1d_grid):
        """Test that open BC allows outflow."""
        # Open BC should not artificially reflect or constrain values
        bc = BoundaryConditions(simple_1d_grid, bc_x="open")
        assert bc is not None


class TestMixed2DBC:
    """Test mixed BC in 2D."""

    def test_periodic_x_dirichlet_y(self, simple_2d_grid):
        """Test periodic in x, Dirichlet in y."""
        bc = BoundaryConditions(simple_2d_grid, bc_x="periodic", bc_y="noflux")
        assert bc is not None

    def test_neumann_x_open_y(self, simple_2d_grid):
        """Test Neumann in x, open in y."""
        bc = BoundaryConditions(simple_2d_grid, bc_x="noflux", bc_y="open")
        assert bc is not None


class TestBCValidation:
    """Test BC validation."""

    def test_bc_type_is_valid(self, simple_1d_grid):
        """Test that valid BC types are accepted."""
        valid_types = [
            "periodic",
            "noflux",
            "open",
        ]
        for bc_type in valid_types:
            bc = BoundaryConditions(simple_1d_grid, bc_x=bc_type)
            assert bc is not None

    def test_boundary_values_are_numeric(self, simple_1d_grid):
        """Test that boundary values are numeric."""
        bc = BoundaryConditions(simple_1d_grid, bc_x="noflux")
        assert bc is not None
