import numpy as np
import plotly.graph_objects as go

# Load the depth map from the .npy file
depth_map = np.load('output/processed/normal_pair4_png.rf.fa99eaa222e8d4acfcfb6483600dda01_segmented_depth.npy')

# Create a meshgrid for X and Y coordinates (width and height of the depth map)
h, w = depth_map.shape
xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))

# Create a 3D surface plot using Plotly
fig = go.Figure(data=[go.Surface(z=depth_map, x=xx, y=yy, colorscale='Viridis')])

# Adjust z-axis range and aspect ratio
fig.update_layout(title='3D Surface Reconstruction from Depth Map',
    scene=dict(xaxis_title='X (Pixels)', yaxis_title='Y (Pixels)', zaxis_title='Depth',
               zaxis=dict(range=[0, 10])))  # Adjust z-axis range as needed
fig.update_scenes(aspectratio=dict(x=1, y=1, z=0.2))  # Adjust aspect ratio to make the plot less tall

# Save the plot to an HTML file
fig.write_html('3d_surface_reconstruction.html')

print("Visualization saved as '3d_surface_reconstruction.html'.")
