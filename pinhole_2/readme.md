# OKAY BITCHES TIME TO DEV THIS SHIT SORRY I'M JUST DOING THIS NOW
## date is October 14, 2023

so now first I'll be trying to develop the loading of data
- get the segmentation points 
- get the RGB depth image
  - needs to be upscaled so that the segmentation points match
- overlay the segmentation points to the depth image

### Cloud Plotting
- do the cloud Plotting

-- refer to the notes --


1. Problem Overview:
   We're working on estimating the volume of objects (in this case, crumpled paper balls) on a plate using a combination of RGB and depth imaging. The challenge is that we have a high-resolution RGB image but only a noisy, low-resolution depth image.

2. Data Sources:
   - High-resolution RGB image of a plate with crumpled paper balls
   - Low-resolution, noisy depth image of the same scene

3. Proposed Approach:
   a) Segmentation:
      - Use Mask R-CNN (or another segmentation method) on the RGB image to precisely identify the plate and paper balls.
   
   b) Depth Image Processing:
      - Scale the low-resolution depth image to match the RGB image's resolution.
      - Apply noise reduction techniques to improve depth data quality.
   
   c) Data Fusion:
      - Overlay the segmentation masks from the RGB image onto the scaled depth image.
      - Extract depth values only for pixels corresponding to the segmented objects.

   d) 3D Reconstruction:
      - Use the camera's intrinsic parameters (from the pinhole camera model) to convert 2D pixel coordinates and depth values into 3D points.
      - The known plate diameter serves as a reference for real-world scaling.

   e) Volume Estimation:
      - Use methods like Convex Hull or more advanced techniques to estimate the volume of the reconstructed 3D points for each paper ball.

4. Key Considerations:
   - Careful alignment of RGB and depth data
   - Handling noise and inaccuracies in the depth data
   - Accounting for potential loss of fine details due to initial low depth resolution
   - Calibration and real-world scaling using the plate as a reference object

5. Potential Enhancements:
   - Multi-view analysis if multiple images are available
   - Implementing more sophisticated depth refinement techniques
   - Exploring machine learning approaches for improved depth estimation

6. Validation:
   - Compare results with ground truth volumes (if available)
   - Analyze performance across different object shapes and sizes

