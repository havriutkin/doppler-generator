import sympy as sp

"""
 In this file, we define all the symbols and equations for the problem
"""

# ========== Define the symbols ==========
f = sp.Symbol('f')  # Frequency of the transmitter
c = sp.Symbol('c')  # Propagation speed
x, y, z = sp.symbols('x y z')   # Position of the transmitter
v_x, v_y, v_z = sp.symbols('v_x v_y v_z')   # Velocity of the transmitter
fi, xi, yi, zi = sp.symbols('f_i x_i y_i z_i')  # Frequency and position of the receiver
vi_x, vi_y, vi_z = sp.symbols('v_i_x v_i_y v_i_z')  # Velocity of the receiver

r_2d = sp.Matrix([x, y])
v_2d = sp.Matrix([v_x, v_y])
r_3d = sp.Matrix([x, y, z])
v_3d = sp.Matrix([v_x, v_y, v_z])

ri_2d = sp.Matrix([xi, yi])
vi_2d = sp.Matrix([vi_x, vi_y])
ri_3d = sp.Matrix([xi, yi, zi])
vi_3d = sp.Matrix([vi_x, vi_y, vi_z])

# ========== Define the equations ==========
doppler_eq_2d = c**2 * (f - fi)**2 * (r_2d - ri_2d).norm()**2 - f**2 * (v_2d.dot(r_2d - ri_2d))**2
doppler_eq_2d = sp.simplify(doppler_eq_2d)

doppler_eq_3d = c**2 * (f - fi)**2 * (r_3d - ri_3d).norm()**2 - f**2 * (v_3d.dot(r_3d - ri_3d))**2
doppler_eq_3d = sp.simplify(doppler_eq_3d)

# TODO: Check if this is correct
observer_freq_2d = f * (1 - (v_2d.dot(r_2d - ri_2d)) / (c * (r_2d - ri_2d).norm()))
observer_freq_3d = f * (1 - (v_3d.dot(r_3d - ri_3d)) / (c * (r_3d - ri_3d).norm()))