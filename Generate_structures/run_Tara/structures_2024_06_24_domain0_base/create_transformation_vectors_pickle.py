## Code to create the transformation vector pickle file.

import numpy as np
import pickle

## Modify the dictionary to make the supercells biggers
# Define the dictionary
transformation_vectors = {
    '100': np.array([[ 2, -2,  2], [ 2,  2, -2], [-6,  6,  6]]),
    '110': np.array([[ 16,  16, -16], [-16,  16,  0], [ 0,  0,  16]]),
    '111': np.array([[-4,  0,  4], [ 2, -4,  2], [ 4,  4,  4]]),
    '210': np.array([[ 2,  2, -2], [-6,  6, -2], [-4,  4, 12]]),
    '211': np.array([[-4,  4,  0], [-2, -2,  6], [ 4,  4,  0]]),
    '221': np.array([[ 0, -4,  4], [ 6, -4, -4], [12,  4,  4]])
}

# Save the dictionary as a pickle file
with open('Transformation_Vectors.pickle', 'wb') as f:
    pickle.dump(transformation_vectors, f)


########## History of the supercells created###
### Original
# transformation_vectors = {
#     '100': np.array([[ 2, -2,  2], [ 2,  2, -2], [-6,  6,  6]]),
#     '110': np.array([[ 2,  2, -2], [-4,  4,  0], [ 0,  0,  8]]),
#     '111': np.array([[-4,  0,  4], [ 2, -4,  2], [ 4,  4,  4]]),
#     '210': np.array([[ 2,  2, -2], [-6,  6, -2], [-4,  4, 12]]),
#     '211': np.array([[-4,  4,  0], [-2, -2,  6], [ 4,  4,  0]]),
#     '221': np.array([[ 0, -4,  4], [ 6, -4, -4], [12,  4,  4]])
# }
