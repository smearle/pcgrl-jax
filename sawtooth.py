import jax.numpy as jnp

def triangle_wave(x, x_peak=10.0, period=1.0):
    # Finding the midpoint
    midpoint = period / 2.0

    # Shift the x-values so that the first peak occurs at x_peak
    x = x - (x_peak - midpoint)

    # Normalizing x to be within a period
    x_normalized = jnp.mod(x, period)
    
    
    # Constructing the rising and falling segments
    rising_segment = x_normalized / midpoint
    falling_segment = 2 - x_normalized / midpoint
    
    # Combining both segments to form a triangle wave
    tri = jnp.where(x_normalized < midpoint, rising_segment, falling_segment)
    
    return tri

# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x_values = jnp.linspace(0, 64, 1000)  # Generate 1000 points between 0 and 2
    y_values = triangle_wave(x_values, period=32.0)  # Generate the y-values using the triangle_wave function

    plt.plot(x_values, y_values)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Triangle Wave')
    plt.savefig('triangle_wave.png')
