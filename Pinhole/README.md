# Plan

### Okay here's what I'm going to do to develop the `Volume estimation` part of the system

- Detect the segmentation and mask of the food using Mask RCNN
- Get the depth Image (RGB-D)
- Reconstruct the 3d shape of the food using the pinhole camera model
- Estimate the volume using the 3d shape (convex hull or voxelization)
- Refine the estimate; This is where the machine learning part comes in (Regression Model). Basta it ain't deep learning
- Test the system.





