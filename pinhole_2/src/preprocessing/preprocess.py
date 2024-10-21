import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.utils.utils import (
    load_rgbd_image,
    load_coco_data,
    create_segmentation_mask,
    get_corresponding_rgbd_filename,
    ensure_directory_exists
)
from src.preprocessing.image_scaling import align_segmentation_mask
from src.utils.visualization import visualize_preprocessing_steps

class PreprocessingPipeline:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.image_input_dir = os.path.join(data_dir, 'image_input')
        self.train_dir = os.path.join(data_dir, 'train_1')
        self.segmentation_file = os.path.join(data_dir, 'train_1', '_annotations.coco.json')
        
        ensure_directory_exists(output_dir)
        self.processed_dir = os.path.join(output_dir, 'processed')
        ensure_directory_exists(self.processed_dir)
        
        self.coco_data = load_coco_data(self.segmentation_file)
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        self.camera_height = 33  # cm
        self.plate_height = 1.5  # cm
        self.plate_depth = 0.7  # cm
        self.plate_category_name = 'plate'  # Adjust this to match your COCO category name for plates

    def find_plate_id(self, segmentation_mask):
        plate_category_id = next((cat['id'] for cat in self.coco_data['categories'] if cat['name'] == self.plate_category_name), None)
        if plate_category_id is None:
            raise ValueError(f"No category found with name '{self.plate_category_name}'")
        
        unique_ids = np.unique(segmentation_mask)
        print(f"Unique IDs in segmentation mask: {unique_ids}")
        print(f"Plate category ID: {plate_category_id}")
        
        if plate_category_id in unique_ids:
            return plate_category_id
        else:
            raise ValueError(f"No plate object found in the segmentation mask. Unique IDs: {unique_ids}")
    def equalize_object_histogram(self, obj_intensity):
        hist, bins = np.histogram(obj_intensity.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        return np.interp(obj_intensity, bins[:-1], cdf_normalized)

    def adaptive_bilateral_filter(self, depth, intensity):
        sigma_color = np.std(intensity) * 0.1
        sigma_space = min(depth.shape) * 0.02
        return cv2.bilateralFilter(depth.astype(np.float32), -1, sigma_color, sigma_space)

    def estimate_relative_depth(self, obj_mask, plate_mask, segmentation_mask):
        obj_centroid = np.mean(np.argwhere(obj_mask), axis=0)
        plate_centroid = np.mean(np.argwhere(plate_mask), axis=0)
        distance = np.linalg.norm(obj_centroid - plate_centroid)
        max_distance = np.sqrt(plate_mask.shape[0]**2 + plate_mask.shape[1]**2)
        
        occlusion_factor = 1 - np.sum(segmentation_mask > 0) / segmentation_mask.size
        
        return (1 - (distance / max_distance)) * (1 + occlusion_factor)

    def depth_from_gradient(self, intensity):
        gradient_x = cv2.Sobel(intensity, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(intensity, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        return 1 - (gradient_mag / np.max(gradient_mag))

    def refine_depth(self, depth, iterations=5):
        for _ in range(iterations):
            depth_smooth = gaussian_filter(depth, sigma=1)
            depth = np.where(np.abs(depth - depth_smooth) > np.std(depth), depth_smooth, depth)
        return depth

    def non_linear_normalize(self, values):
        min_val, max_val = np.min(values), np.max(values)
        normalized = (values - min_val) / (max_val - min_val)
        return np.power(normalized, 0.7)  # Adjust exponent as needed

    def apply_depth_consistency(self, depth_map, segmentation_mask):
        for obj_id in np.unique(segmentation_mask):
            if obj_id == 0:  # Skip background
                continue
            obj_mask = segmentation_mask == obj_id
            obj_depth = depth_map[obj_mask]
            
            q1, q3 = np.percentile(obj_depth, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (obj_depth < lower_bound) | (obj_depth > upper_bound)
            
            local_median = median_filter(obj_depth, size=5)
            obj_depth[outlier_mask] = local_median[outlier_mask]
            
            depth_map[obj_mask] = obj_depth
        
        return depth_map

    def color_to_depth(self, rgbd_image, segmentation_mask):
        rgb = rgbd_image[:,:,:3].astype(float)
        intensity = np.dot(rgb, [0.299, 0.587, 0.114])
        depth = np.zeros_like(intensity)
        
        try:
            plate_id = self.find_plate_id(segmentation_mask)
            plate_mask = segmentation_mask == plate_id
            depth[plate_mask] = self.plate_height + self.plate_depth/2
            
            largest_object_size = max(np.sum(segmentation_mask == obj_id) for obj_id in np.unique(segmentation_mask) if obj_id != 0 and obj_id != plate_id)
            max_object_height = min(15, self.plate_height + (largest_object_size / (rgbd_image.shape[0] * rgbd_image.shape[1])) * 30)
            
            for obj_id in np.unique(segmentation_mask):
                if obj_id == 0 or obj_id == plate_id:
                    continue
                
                obj_mask = segmentation_mask == obj_id
                obj_intensity = intensity[obj_mask]
                
                obj_intensity = self.equalize_object_histogram(obj_intensity)
                obj_norm_intensity = self.non_linear_normalize(obj_intensity)
                obj_inv_intensity = 1 - obj_norm_intensity
                
                min_object_height = 0.5
                obj_depth = obj_inv_intensity * (max_object_height - min_object_height) + min_object_height
                
                obj_depth_2d = np.zeros_like(depth)
                obj_depth_2d[obj_mask] = obj_depth
                
                obj_depth_filtered = self.adaptive_bilateral_filter(obj_depth_2d, intensity)
                
                relative_depth = self.estimate_relative_depth(obj_mask, plate_mask, segmentation_mask)
                obj_depth_filtered[obj_mask] *= relative_depth
                
                gradient_depth = self.depth_from_gradient(intensity)
                obj_depth_filtered[obj_mask] = (obj_depth_filtered[obj_mask] + gradient_depth[obj_mask]) / 2
                
                obj_depth_filtered[obj_mask] += self.plate_height + self.plate_depth
                
                depth[obj_mask] = obj_depth_filtered[obj_mask]
            
            background_mask = segmentation_mask == 0
            depth[background_mask] = self.camera_height
            
            depth = self.refine_depth(depth)
            depth = self.apply_depth_consistency(depth, segmentation_mask)
        
        except Exception as e:
            print(f"Error in color to depth conversion: {str(e)}")
            depth.fill(self.camera_height)  # Set all depths to camera height as a fallback
        
        return depth

    def analyze_object_depths(self, depth_map, segmentation_mask):
        object_stats = {}
        unique_objects = np.unique(segmentation_mask)
        unique_objects = unique_objects[unique_objects != 0]  # Exclude background

        for obj_id in unique_objects:
            obj_mask = segmentation_mask == obj_id
            obj_depths = depth_map[obj_mask]
            
            object_stats[obj_id] = {
                'min_depth': np.min(obj_depths),
                'max_depth': np.max(obj_depths),
                'mean_depth': np.mean(obj_depths),
                'median_depth': np.median(obj_depths),
                'std_depth': np.std(obj_depths)
            }

        return object_stats

    def visualize_object_depths(self, rgb_image, depth_map, segmentation_mask, object_stats, output_path):
        num_objects = len(object_stats)
        fig, axs = plt.subplots(2, 3, figsize=(20, 10 * (num_objects // 3 + 1)))
        fig.suptitle('Object Depth Visualization', fontsize=16)

        axs[0, 0].imshow(rgb_image)
        axs[0, 0].set_title('Original RGB Image')
        axs[0, 0].axis('off')

        im = axs[0, 1].imshow(depth_map, cmap='viridis')
        axs[0, 1].set_title('Full Depth Map')
        axs[0, 1].axis('off')
        plt.colorbar(im, ax=axs[0, 1], label='Depth (cm)')

        axs[0, 2].imshow(segmentation_mask, cmap='tab20')
        axs[0, 2].set_title('Segmentation Mask')
        axs[0, 2].axis('off')

        for i, (obj_id, stats) in enumerate(object_stats.items()):
            row = (i + 3) // 3
            col = (i + 3) % 3
            obj_mask = segmentation_mask == obj_id
            obj_depth = np.ma.masked_where(~obj_mask, depth_map)
            im = axs[row, col].imshow(obj_depth, cmap='viridis')
            axs[row, col].set_title(f'Object {obj_id} Depth')
            axs[row, col].axis('off')
            plt.colorbar(im, ax=axs[row, col], label='Depth (cm)')
            
            stats_text = f"Min: {stats['min_depth']:.2f}\nMax: {stats['max_depth']:.2f}\n"
            stats_text += f"Mean: {stats['mean_depth']:.2f}\nMedian: {stats['median_depth']:.2f}\n"
            stats_text += f"Std: {stats['std_depth']:.2f}"
            axs[row, col].text(0.05, 0.95, stats_text, transform=axs[row, col].transAxes, 
                               verticalalignment='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def upscale_with_padding(self, image, target_shape):
        h, w = image.shape[:2]
        target_h, target_w = target_shape
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=resized.dtype)
        else:
            padded = np.zeros((target_h, target_w), dtype=resized.dtype)
        
        pad_h, pad_w = (target_h - new_h) // 2, (target_w - new_w) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        return padded

    def normalize_plate_color(self, upscaled_rgbd, segmentation_mask):
        try:
            plate_id = self.find_plate_id(segmentation_mask)
            plate_mask = segmentation_mask == plate_id
            plate_colors = upscaled_rgbd[plate_mask][:, :3]
            
            print(f"Plate ID: {plate_id}")
            print(f"Plate mask shape: {plate_mask.shape}")
            print(f"Plate mask sum: {np.sum(plate_mask)}")
            print(f"Plate colors shape: {plate_colors.shape}")
            
            if plate_colors.shape[0] == 0:
                raise ValueError("No plate pixels found in the image.")
            
            # Use a simple average color if there are too few plate pixels
            if plate_colors.shape[0] < 10:
                normalized_plate_color = np.mean(plate_colors, axis=0)
            else:
                kmeans = KMeans(n_clusters=min(3, plate_colors.shape[0]), random_state=42)
                kmeans.fit(plate_colors)
                
                cluster_sizes = np.bincount(kmeans.labels_)
                sorted_clusters = sorted(zip(cluster_sizes, kmeans.cluster_centers_), reverse=True)
                dominant_color = sorted_clusters[0][1]
                
                color_std = np.std(plate_colors, axis=0)
                distance_threshold = np.mean(color_std) * 2
                
                close_to_dominant = np.linalg.norm(plate_colors - dominant_color, axis=1) < distance_threshold
                normalized_plate_color = np.mean(plate_colors[close_to_dominant], axis=0)
            
            print(f"Normalized plate color: {normalized_plate_color}")
        except Exception as e:
            print(f"Error in color normalization: {str(e)}")
            print("Using default normalization.")
            normalized_plate_color = np.array([128, 128, 128])  # Default gray color
        
        normalized_rgbd = upscaled_rgbd.copy().astype(float)
        for i in range(3):
            normalized_rgbd[:,:,i] = (normalized_rgbd[:,:,i] - normalized_plate_color[i]) / normalized_plate_color[i] * 0.7 + 0.7
        
        return normalized_rgbd, normalized_plate_color
    def process_image(self, rgb_filename, image_id):
        try:
            rgb_path = os.path.join(self.train_dir, rgb_filename)
            rgb_image = cv2.imread(rgb_path)
            if rgb_image is None:
                raise FileNotFoundError(f"Could not load RGB image at {rgb_path}")
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            rgbd_filename = get_corresponding_rgbd_filename(rgb_filename)
            if rgbd_filename is None:
                raise ValueError(f"No corresponding RGBD filename found for {rgb_filename}")

            rgbd_path = os.path.join(self.image_input_dir, rgbd_filename)
            rgbd_image = load_rgbd_image(rgbd_path)

            upscaled_rgbd = self.upscale_with_padding(rgbd_image, rgb_image.shape[:2])

            segmentation_mask = create_segmentation_mask(image_id, self.coco_data)
            segmentation_mask = align_segmentation_mask(segmentation_mask, rgb_image.shape[:2])
            
            print(f"Segmentation mask shape: {segmentation_mask.shape}")
            print(f"Unique segmentation values: {np.unique(segmentation_mask)}")
            
            # Extract and calibrate depth from RGBD
            calibrated_depth = self.color_to_depth(upscaled_rgbd, segmentation_mask)
            
            # Analyze object depths
            object_stats = self.analyze_object_depths(calibrated_depth, segmentation_mask)

            # Normalize RGB channels (if needed)
            try:
                normalized_rgbd, normalized_plate_color = self.normalize_plate_color(upscaled_rgbd, segmentation_mask)
            except Exception as e:
                print(f"Error in plate color normalization: {str(e)}")
                print("Using non-normalized RGBD data.")
                normalized_rgbd = upscaled_rgbd
                normalized_plate_color = np.array([128, 128, 128])  # Default gray color

            # Prepare segmented data
            segmented_data = {}
            for obj_id in np.unique(segmentation_mask):
                if obj_id == 0:  # Assuming 0 is background
                    continue
                object_mask = segmentation_mask == obj_id
                
                segmented_data[obj_id] = {
                    'rgb': rgb_image.copy() * object_mask[:,:,np.newaxis],
                    'rgbd': normalized_rgbd.copy() * object_mask[:,:,np.newaxis],
                    'depth': calibrated_depth.copy() * object_mask,
                    'colors': normalized_rgbd[object_mask],
                    'name': self.category_id_to_name.get(obj_id, f"unknown_object_{obj_id}")
                }

            # Visualize preprocessing steps including depth
            vis_filename = f"{os.path.splitext(rgb_filename)[0]}_visualization.png"
            visualize_preprocessing_steps(
                rgb_image, rgbd_image, normalized_rgbd, segmentation_mask, 
                segmented_data, self.output_dir, vis_filename, 
                normalized_plate_color, calibrated_depth
            )

            return rgb_image, normalized_rgbd, calibrated_depth, segmentation_mask, segmented_data, normalized_plate_color, object_stats
        except Exception as e:
            print(f"Error processing image {rgb_filename}: {str(e)}")
            raise
    def run(self):
        rgb_filenames = [
            'normal_pair4_png.rf.fa99eaa222e8d4acfcfb6483600dda01.jpg'
        ]
        image_ids = [0]  # Corresponding image IDs in the COCO dataset

        for rgb_filename, image_id in zip(rgb_filenames, image_ids):
            print(f"Processing {rgb_filename}")
            rgb, normalized_rgbd, depth, mask, segmented_data, normalized_plate_color, object_stats = self.process_image(rgb_filename, image_id)
            print(f"Processed {rgb_filename}.")
            print(f"RGB shape: {rgb.shape}, Normalized RGBD shape: {normalized_rgbd.shape}, Depth shape: {depth.shape}, Mask shape: {mask.shape}")
            print(f"Number of segmented objects: {len(segmented_data)}")
            print(f"Normalized plate color: {normalized_plate_color}")
            print("Object depth statistics:")
            for obj_id, stats in object_stats.items():
                print(f"  Object {obj_id} ({self.category_id_to_name.get(obj_id, 'Unknown')}):")
                for stat_name, stat_value in stats.items():
                    print(f"    {stat_name}: {stat_value:.2f} cm")
            print(f"Saved processed data to {self.processed_dir}")
            print("---")
