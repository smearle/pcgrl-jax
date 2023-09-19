import jax
import jax.numpy as jnp
from jax import random

# Define convolution kernel for identifying valid triplets
vert_kern = jnp.array([[1, 0, 0],
                       [-1, 0, 0],
                       [-1, 0, 0]],
                      jnp.float32)
hor_kern = jnp.array([[1, -1, -1],
                      [0, 0, 0],
                      [0, 0, 0]],
                     jnp.float32)
conv_kernel = jnp.stack([vert_kern, hor_kern])[:,None]

def generate_maze(grid_shape):
    # Initialize the random number generator
    key = random.PRNGKey(0)
    
    # Create the initial grid with all walls
    grid = jnp.zeros(grid_shape, dtype=jnp.float32)[None,None]
    
    # Choose a random initial position and set it to an empty tile
    init_position = random.randint(key, (2,), 0, jnp.array(grid_shape))
    # grid = jax.ops.index_update(grid, jax.ops.index[init_position[0], init_position[1]], 1)
    grid = grid.at[0, 0, init_position[0], init_position[1]].set(1)
    
    # Function to identify valid triplets and replace them
    def update_grid(grid):
        # Use convolution to find valid triplets (empty, wall, wall)
        valid_triplets = jax.lax.conv(grid, conv_kernel, (1, 1), 'SAME')
        
        # Find the indices of valid triplets
        triplet_indices = jnp.argwhere(valid_triplets == 1)
        breakpoint()
        
        if triplet_indices.shape[0] == 0:
            return grid, False  # No more valid triplets found, exit
        
        # Randomly choose one of the valid triplets
        rand_index = random.randint(key, (1,), 0, triplet_indices.shape[0])[0]
        triplet = triplet_indices[rand_index]
        
        # Replace the wall tiles with empty tiles in the selected triplet
        # grid = jax.ops.index_update(grid, jax.ops.index[triplet[0], triplet[1]], 1)
        grid = grid.at[triplet[0], triplet[1]].set(1)
        # grid = jax.ops.index_update(grid, jax.ops.index[triplet[0], triplet[1]+1], 1)
        grid = grid.at[triplet[0], triplet[1]+1].set(1)
        
        return grid, True
    
    # Generate the maze until no more valid triplets are found
    while True:
        grid, found_triplets = update_grid(grid)
        if not found_triplets:
            break
    
    return grid

# Example usage:
grid_shape = (10, 10)  # Change the shape as needed
maze = generate_maze(grid_shape)
print(maze)