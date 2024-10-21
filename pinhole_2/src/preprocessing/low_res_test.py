# src/preprocessing/test_low_res_rgbd.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.utils.utils import load_rgbd_image, get_corresponding_rgbd_filename
from src.preprocessing.noise_reduction import reduce_depth_noise

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
def visualize_rgbd(rgbd_image, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    #  RGB real image
    ax1.imshow(rgbd_image[:,:,:3])
    ax1.set_title("RGB")
    ax1.axis('off')
    
    # Depth factors
    depth_vis = ax2.imshow(rgbd_image[:,:,3], cmap='jet')
    ax2.set_title("Depth")
    ax2.axis('off')
    plt.colorbar(depth_vis, ax=ax2, label='Depth')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def process_low_res_rgbd(rgb_filename, output_dir):
    # get corresponding RGBD filename
    rgbd_filename = get_corresponding_rgbd_filename(rgb_filename)
    if rgbd_filename is None:
        raise ValueError(f"No corresponding RGBD filename found for {rgb_filename}")

    # load low-resolution RGBD image
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    image_input_dir = os.path.join(data_dir, 'image_input')
    rgbd_path = os.path.join(image_input_dir, rgbd_filename)
    original_rgbd = load_rgbd_image(rgbd_path)

    print(f"Loaded RGBD image shape: {original_rgbd.shape}")
    print(f"Depth channel min: {original_rgbd[:,:,3].min()}, max: {original_rgbd[:,:,3].max()}")
    print(f"Unique depth values: {np.unique(original_rgbd[:,:,3])}")

    #vbisualize original RGBD
    fig_original = visualize_rgbd(original_rgbd, "Original Low-res RGBD")
    fig_original.savefig(os.path.join(output_dir, f"{os.path.splitext(rgbd_filename)[0]}_original.png"))
    plt.close(fig_original)

    # Apply noise reduction to depth channel
    depth = original_rgbd[:,:,3].astype(np.float32)
    noise_reduced_depth = reduce_depth_noise(depth)

    print(f"Noise reduced depth min: {noise_reduced_depth.min()}, max: {noise_reduced_depth.max()}")

    # create noise-reduced RGBD image
    noise_reduced_rgbd = original_rgbd.copy()
    noise_reduced_rgbd[:,:,3] = noise_reduced_depth

    # Visualize noise-reduced RGBD
    fig_reduced = visualize_rgbd(noise_reduced_rgbd, "Noise-reduced Low-res RGBD")
    fig_reduced.savefig(os.path.join(output_dir, f"{os.path.splitext(rgbd_filename)[0]}_noise_reduced.png"))
    plt.close(fig_reduced)

    return original_rgbd, noise_reduced_rgbd

def main():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'low_res_test'))
    os.makedirs(output_dir, exist_ok=True)

    rgb_filenames = [
        'normal_pair4_png.rf.fa99eaa222e8d4acfcfb6483600dda01.jpg'
    ]

    for rgb_filename in rgb_filenames:
        print(f"Processing {rgb_filename}")
        original_rgbd, noise_reduced_rgbd = process_low_res_rgbd(rgb_filename, output_dir)
        print("---")

if __name__ == "__main__":
    main()
