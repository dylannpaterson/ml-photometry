# Model-dependent constants for the Roman Point Source ML Pipeline
# Shared between Castor (Training) and Pollux (Production)

DEFAULT_CELL_SIZE = 4
MAX_CAPACITY_PER_CELL = 3
# Enforce a 4x Upscale-Safe Odd Size (129 = 4 * 32 + 1). 
# This guarantees exact convolution centers (idx 64) and prevents 0.5 pixel parity shifts.
SHAPE_SIZE = 129
N_PCA_COMPONENTS = 20 # Used for Physics Prior Init (No longer in output grid)
GLOBAL_STRETCH_SCALE = 10.0
