# Autonomous Drone Racing Project Course

## Feature Wishlist
- Controller callback after simulation finished
  - This is useful to do evaluate relevant controller stats, plot some graphs, ...
- Pass information about gate/obstacle sizes in initial_info (or at some later point)
  - This is necessary to calculate gate/obstacle hit boxes
- (Nominal) drone properties in initial info dict
  - Reduces chance of parameter mismatch when using model based approaches like MPC
- 