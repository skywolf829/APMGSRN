Note for NGP models:

To calculate the number of grid parameters for NGP, we would do:
n_features * n_grids * hash_log2_size

However, since the grids at the coarsest level do not have a number of vertices as large as hash_log2_size, 
there are many "unused" table entries for that level, that are discarded since they aren't used.
Effectively, this makes the saved NGP model smaller than the original calculation above. Therefore, for a more fair
comparison, we write down some heuristics here that we find give a saved model size closer to the expected
size. 

Each of these assume the NGP model has n_features=2. Other parameters are adjusted to get the 
total model size as similar as possible to our method and fVSRN for comparison.

Goal # params       hash_base_resolution        hash_max_resolution     n_grids     hash_log2_size
2^16                8                           256                     17          11
2^20                16                          512                     18          15
2^24                16                          256                     17          20
