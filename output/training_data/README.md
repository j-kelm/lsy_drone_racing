This folder should contain the `.npz` files with the processed training data.

The training data contains four fields:
- `actions`: Actions like accepted by the env
- `local_actions:` Actions in the local feature space
- `obs`: Observations like obtained from the env
- `local_obs`: Observations in the local feature space

The local components are used for training, while keeping the unmodified equivalents around is handy for plotting and comparing actions.
The index is consistent between all groups.