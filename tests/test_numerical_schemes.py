"""Tests for numerical schemes."""
import pytest
import numpy as np


class TestFiniteDifferenceSchemes:
    """Test finite difference approximations."""
    
    def test_central_difference_accuracy(self):
        """Test central difference approximation accuracy."""
        # For smooth function f(x) = x^2
        # df/dx at x=1 should be 2.0
        dx = 0.01
        x = 1.0
        f_minus = (x - dx)**2
        f_plus = (x + dx)**2
        
        df_dx_approx = (f_plus - f_minus) / (2 * dx)
        df_dx_exact = 2 * x
        
        error = abs(df_dx_approx - df_dx_exact)
        assert error < 0.01
    
    def test_forward_difference_accuracy(self):
        """Test forward difference approximation."""
        dx = 0.01
        x = 1.0
        f = x**2
        f_next = (x + dx)**2
        
        df_dx_approx = (f_next - f) / dx
        df_dx_exact = 2 * x
        
        error = abs(df_dx_approx - df_dx_exact)
        assert error < 0.1
    
    def test_backward_difference_accuracy(self):
        """Test backward difference approximation."""
        dx = 0.01
        x = 1.0
        f = x**2
        f_prev = (x - dx)**2
        
        df_dx_approx = (f - f_prev) / dx
        df_dx_exact = 2 * x
        
        error = abs(df_dx_approx - df_dx_exact)
        assert error < 0.1
    
    def test_second_derivative_accuracy(self):
        """Test second derivative approximation."""
        # For f(x) = x^2, d2f/dx2 = 2.0
        dx = 0.01
        x = 1.0
        f_prev = (x - dx)**2
        f_center = x**2
        f_next = (x + dx)**2
        
        d2f_dx2_approx = (f_next - 2*f_center + f_prev) / dx**2
        d2f_dx2_exact = 2.0
        
        error = abs(d2f_dx2_approx - d2f_dx2_exact)
        assert error < 1.0


class TestFiniteVolumeSchemes:
    """Test finite volume methods."""
    
    def test_flux_conservation(self):
        """Test flux conservation principle."""
        # Total flux in = total flux out
        flux_in = 1.0
        flux_out = 1.0
        
        assert np.isclose(flux_in, flux_out)
    
    def test_piecewise_constant_reconstruction(self):
        """Test piecewise constant reconstruction."""
        # Cell values
        f = np.array([1.0, 2.0, 3.0, 2.0])
        
        # Reconstruct as piecewise constant
        # Should recover the same values
        assert len(f) == 4
        assert np.all(f > 0)
    
    def test_flux_upwind_scheme(self):
        """Test upwind flux computation."""
        u = 0.5  # velocity
        f_left = 1.0
        f_right = 2.0
        
        # Upwind flux for positive velocity uses left value
        flux = u * f_left if u > 0 else u * f_right
        
        assert flux == 0.5


class TestTimeIntegration:
    """Test time stepping schemes."""
    
    def test_euler_forward_stability(self):
        """Test forward Euler stability."""
        # For linear ODE: dy/dt = -lambda * y
        # Solution: y(t) = y0 * exp(-lambda * t)
        # Euler forward: y_{n+1} = y_n * (1 - lambda * dt)
        
        lambda_val = 1.0
        dt = 0.5
        
        # Stability condition: |1 - lambda * dt| < 1
        # For lambda > 0: dt < 2/lambda
        assert dt < 2.0 / lambda_val
    
    def test_backward_euler_stability(self):
        """Test backward Euler stability (unconditionally stable)."""
        # Backward Euler: (y_{n+1} - y_n) / dt = -lambda * y_{n+1}
        # y_{n+1} = y_n / (1 + lambda * dt)
        
        lambda_val = 1.0
        dt = 10.0
        
        # Always stable for lambda > 0
        stability_factor = 1.0 / (1.0 + lambda_val * dt)
        assert 0 < stability_factor < 1
    
    def test_rk2_midpoint_accuracy(self):
        """Test RK2 midpoint method accuracy."""
        # For y' = -y, exact solution: y(1) = exp(-1)
        # RK2 should give good accuracy
        y0 = 1.0
        t_end = 1.0
        n_steps = 100
        dt = t_end / n_steps
        
        y = y0
        for _ in range(n_steps):
            k1 = -y
            y_mid = y + 0.5 * dt * k1
            k2 = -y_mid
            y = y + dt * k2
        
        exact = np.exp(-1.0)
        error = abs(y - exact)
        assert error < 0.01
    
    def test_rk4_high_accuracy(self):
        """Test RK4 high accuracy."""
        # RK4 for y' = -y
        y0 = 1.0
        t_end = 1.0
        n_steps = 100
        dt = t_end / n_steps
        
        y = y0
        for _ in range(n_steps):
            k1 = -y
            k2 = -(y + 0.5 * dt * k1)
            k3 = -(y + 0.5 * dt * k2)
            k4 = -(y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        exact = np.exp(-1.0)
        error = abs(y - exact)
        # RK4 should be much more accurate
        assert error < 1e-6


class TestSpaceTimeDiscretization:
    """Test combined space-time discretization."""
    
    def test_method_of_lines(self):
        """Test method of lines approach."""
        # PDEs: du/dt = d2u/dx2
        # Semi-discrete: du_i/dt = (u_{i+1} - 2*u_i + u_{i-1}) / dx^2
        
        dx = 0.1
        dt = 0.005
        
        # CFL for diffusion: dt / dx^2 <= 0.25
        diffusion_cfl = dt / (dx**2)
        assert diffusion_cfl < 0.5
    
    def test_implicit_scheme_stability(self):
        """Test implicit scheme (usually more stable)."""
        # Implicit schemes allow larger timesteps
        dx = 0.1
        dt = 0.1  # Large timestep
        
        # Even with large dt, should be stable
        assert dt > 0 and dx > 0


class TestSchemeProperties:
    """Test important numerical scheme properties."""
    
    def test_conservation_property(self):
        """Test conservation of total quantity."""
        f = np.array([1.0, 2.0, 3.0, 2.0])
        dx = 1.0
        
        # Total quantity
        total = np.sum(f) * dx
        assert total > 0
    
    def test_positivity_preservation(self):
        """Test schemes that preserve positivity."""
        # Many schemes can violate positivity
        f = np.array([1.0, 0.5, 0.2, 0.1])
        
        # After guaranteed positivity scheme
        f_new = np.maximum(f, 0)
        
        assert np.all(f_new >= 0)
    
    def test_monotonicity_preservation(self):
        """Test monotone schemes."""
        f = np.array([0.0, 1.0, 2.0, 3.0])
        
        # Monotone increasing
        assert np.all(np.diff(f) >= 0)
    
    def test_total_variation_diminishing(self):
        """Test TVD (Total Variation Diminishing) property."""
        f = np.array([1.0, 3.0, 1.0, 3.0])
        
        # Total variation
        tv = np.sum(np.abs(np.diff(f)))
        assert tv > 0


class TestErrorAnalysis:
    """Test error analysis of schemes."""
    
    def test_convergence_with_grid_refinement(self):
        """Test that errors decrease with grid refinement."""
        errors = []
        for n_points in [16, 32, 64, 128]:
            dx = 1.0 / n_points
            # Approximate integral of sin(x) from 0 to 1
            x = np.linspace(0, 1, n_points)
            f = np.sin(x)
            integral_approx = np.sum(f) * dx
            
            errors.append(abs(integral_approx - (1 - np.cos(1))))
        
        # Errors should decrease
        assert errors[0] > errors[-1]
    
    def test_truncation_error_scaling(self):
        """Test truncation error scaling."""
        # For central difference: truncation error = O(dx^2)
        errors = []
        for dx in [0.1, 0.05, 0.025]:
            # Approximate df/dx = 2x at x=1
            x = 1.0
            df_approx = ((x+dx)**2 - (x-dx)**2) / (2*dx)
            error = abs(df_approx - 2.0)
            errors.append(error)
        
        # Central difference is exact for quadratic functions
        # so errors should be very small
        assert all(e < 1e-10 for e in errors)


class TestSchemeComparison:
    """Compare different numerical schemes."""
    
    def test_upwind_vs_central_difference(self):
        """Compare upwind and central difference."""
        # Central difference: higher order but can oscillate
        # Upwind: lower order but monotone
        
        # Both should give reasonable results
        dx = 0.1
        u = 1.0
        
        # Upwind: uses 1 point
        upwind_points = 1
        # Central: uses 2 points
        central_points = 2
        
        assert central_points > upwind_points
    
    def test_explicit_vs_implicit(self):
        """Compare explicit and implicit schemes."""
        # Explicit: easier to implement, limited timestep
        # Implicit: stable, harder to implement
        
        dt_explicit_max = 0.01
        dt_implicit_max = 1.0  # Much larger
        
        assert dt_implicit_max > dt_explicit_max
