# Extra pip installations:
# pip install opencv-python
# pip install onnxruntime

#%% Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import os
import urllib.request
from hypha_rpc.sync import connect_to_server
from tifffile import imread
from kaibu_utils import mask_to_features


#%% Read example image
image = imread("example_image.tif")
if len(image.shape) == 3 and image.shape[0] == 3:
    image = np.transpose(image, [1, 2, 0])
assert image.shape == (512, 512, 3)

# Save the image to png
cv2.imwrite("example_image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Reverse the color channels to match matplotlib's default color mapping
image = image[..., ::-1]


# %% Connect to SAM service and compute image embedding
if not os.path.exists("example_image_embeddings.bin"):
    server_url = "https://hypha.aicell.io"
    workspace_name = "bioimageio-colab"
    service_id = "microsam"
    model_name = "vit_b_lm"

    client = connect_to_server({"server_url": server_url, "method_timeout": 5})
    sid = f"{workspace_name}/{service_id}"
    svc = client.get_service(sid, {"mode": "random"})
    features = svc.compute_embedding(model_name=model_name, image=image)

    print(features.shape)

    # Save the embeddings to a binary file (.bin)
    features.tofile("example_image_embeddings.bin")
else:
    # Load the embeddings from the binary file
    features = np.fromfile("example_image_embeddings.bin", dtype=np.float32)

    # Reshape the embeddings to the expected shape
    features = features.reshape(1, 256, 64, 64)


# %% Load the SAM decoder model
model_url = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/sam-vit_b_lm-decoder/1/model.onnx"
local_model_path = "model.onnx"

if not os.path.exists(local_model_path):
    urllib.request.urlretrieve(model_url, local_model_path)

model = ort.InferenceSession(local_model_path)

# %%
def model_data(clicks, image_embedding, model_scale):
    # Initialize variables
    point_coords = None
    point_labels = None

    # Check for input click prompts
    if clicks:
        n = len(clicks)

        # Initialize arrays for (n+1) points (including padding point)
        point_coords = np.zeros((n + 1, 2), dtype=np.float32)
        point_labels = np.zeros((n + 1,), dtype=np.float32)

        # Add clicks and scale coordinates
        for i, click in enumerate(clicks):
            point_coords[i, 0] = click["x"] * model_scale["samScale"]
            point_coords[i, 1] = click["y"] * model_scale["samScale"]
            point_labels[i] = click["clickType"]

        # Add the padding point
        point_coords[n, 0] = 0.0
        point_coords[n, 1] = 0.0
        point_labels[n] = -1.0

        # Convert to tensors
        point_coords_tensor = ort.OrtValue.ortvalue_from_numpy(
            np.expand_dims(point_coords, axis=0)
        )
        point_labels_tensor = ort.OrtValue.ortvalue_from_numpy(
            np.expand_dims(point_labels, axis=0)
        )
    else:
        return None

    # Image size tensor
    image_size_tensor = ort.OrtValue.ortvalue_from_numpy(
        np.array([model_scale["height"], model_scale["width"]], dtype=np.float32)
    )

    # Mask input and has_mask_input
    mask_input = ort.OrtValue.ortvalue_from_numpy(
        np.zeros((1, 1, 256, 256), dtype=np.float32)
    )
    has_mask_input = ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.float32))

    return {
        "image_embeddings": image_embedding,
        "point_coords": point_coords_tensor,
        "point_labels": point_labels_tensor,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
        "orig_im_size": image_size_tensor,
    }

# Define inputs
point_coords = [(80, 80)]
clicks = [
    {"x": coord[0], "y": coord[1], "clickType": 1} for coord in point_coords
]
img_height, img_width = image.shape[:2]
model_scale = {
    "height": img_height,  # original image height
    "width": img_width,  # original image width
    "samScale": 1024 / max(img_height, img_width),  # scaling factor
}

# Generate feeds
feeds = model_data(clicks, features, model_scale)

#%% Run the model
output_names = ["masks"]
masks = model.run(output_names, feeds)

# %% Plot the mask
mask = masks[0].squeeze()

# Display the mask using matplotlib
plt.imshow(mask, cmap='viridis')
plt.colorbar(label="Normalized Intensity")
plt.title("Visualized Mask")
plt.axis("off")
plt.show()

# %%

# Convert the mask to a binary mask
binary_mask = (mask > 0).astype(np.uint8)

# Find contours
contours, hierarchy = cv2.findContours(
    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Create an empty image to draw contours
contour_image = np.zeros_like(binary_mask)

# Draw contours on the empty image
cv2.drawContours(contour_image, contours, -1, (255), thickness=1)  # Thickness can be adjusted

# Visualize the contours
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Binary Mask")
plt.imshow(binary_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Contours")
plt.imshow(contour_image, cmap="gray")
plt.axis("off")
plt.show()

# %%
image_overlay = image.copy()

# Draw contours on the RGB image
cv2.drawContours(image_overlay, contours, -1, (255, 0, 255), thickness=2)  # Green contours

# Plot the overlay
plt.figure(figsize=(8, 8))
plt.title("Overlay of Contours on Original Image")
plt.imshow(cv2.cvtColor(image_overlay, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for plotting
plt.axis("off")
plt.show()

# %%
def contours_to_geojson(contours):
    """
    Convert OpenCV contours into a GeoJSON structure.
    
    Parameters:
        contours (list): List of OpenCV contours.
        
    Returns:
        dict: GeoJSON object.
    """
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for contour in contours:
        # Convert contour to a list of [x, y] points
        points = contour[:, 0, :].tolist()
        
        # Ensure the polygon is closed (first point == last point)
        if points[0] != points[-1]:
            points.append(points[0])
        
        # Create a GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [points]
            },
            "properties": {}  # Add any desired properties here
        }
        geojson["features"].append(feature)
    
    return geojson

# Example usage
geojson_data = contours_to_geojson(contours)


# %%
features = mask_to_features(binary_mask)
# %%
