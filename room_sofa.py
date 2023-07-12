import cv2
import os
import numpy as np

# Load the original room, bed, sofa, painting, lamp, chair, and table images
room_image_path = './images/room/room.jpg'  # Path to the blank room image
sofa_image_path = './images/sofa/sofa.png'  # Path to the sofa image
painting_image_path = './images/painting/painting1.png'  # Path to the painting image
lamp_image_path = './images/lamp/lamp1.png'  # Path to the lamp image
# Path to the additional lamp image
additional_lamp_image_path = './images/lamp/lamp2.png'
output_folder = './output/'

room_image = cv2.imread(room_image_path, cv2.IMREAD_COLOR)
sofa_image = cv2.imread(sofa_image_path, cv2.IMREAD_UNCHANGED)
painting_image = cv2.imread(painting_image_path, cv2.IMREAD_UNCHANGED)
lamp_image = cv2.imread(lamp_image_path, cv2.IMREAD_UNCHANGED)
additional_lamp_image = cv2.imread(
    additional_lamp_image_path, cv2.IMREAD_UNCHANGED)

# Define the coordinates to place the bed, sofa, painting, lamp, chair, table, and additional lamp images in the room image
sofa_x_coord = 160  # Update with the desired x-coordinate for the sofa
sofa_y_coord = 430  # Update with the desired y-coordinate for the sofa
painting_x_coord = 360  # Update with the desired x-coordinate for the painting
painting_y_coord = 200  # Update with the desired y-coordinate for the painting
lamp_x_coord = 600  # Update with the desired x-coordinate for the lamp
lamp_y_coord = 140  # Update with the desired y-coordinate for the lamp

# Update with the desired x-coordinate for the additional lamp
additional_lamp_x_coord = 115
# Update with the desired y-coordinate for the additional lamp
additional_lamp_y_coord = 140

# Define the scale factors for resizing the bed, sofa, painting, lamp, chair, table, and additional lamp images
# Adjust the scale factors as desired
sofa_scale_factor = 0.6
painting_scale_factor = 0.2
lamp_scale_factor = 0.3
additional_lamp_scale_factor = 0.3

# Resize the bed, sofa, painting, lamp, chair, table, and additional lamp images with the scale factors
sofa_image_resized = cv2.resize(
    sofa_image, None, fx=sofa_scale_factor, fy=sofa_scale_factor)
painting_image_resized = cv2.resize(
    painting_image, None, fx=painting_scale_factor, fy=painting_scale_factor)
lamp_image_resized = cv2.resize(
    lamp_image, None, fx=lamp_scale_factor, fy=lamp_scale_factor)
additional_lamp_image_resized = cv2.resize(
    additional_lamp_image, None, fx=additional_lamp_scale_factor, fy=additional_lamp_scale_factor)

# Create masks for the bed, sofa, painting, lamp, chair, table, and additional lamp images
if sofa_image_resized.shape[2] == 4:
    sofa_alpha_mask = sofa_image_resized[:, :, 3] / 255.0
else:
    sofa_alpha_mask = np.ones(
        (sofa_image_resized.shape[0], sofa_image_resized.shape[1]))

if painting_image_resized.shape[2] == 4:
    painting_alpha_mask = painting_image_resized[:, :, 3] / 255.0
else:
    painting_alpha_mask = np.ones(
        (painting_image_resized.shape[0], painting_image_resized.shape[1]))

if lamp_image_resized.shape[2] == 4:
    lamp_alpha_mask = lamp_image_resized[:, :, 3] / 255.0
else:
    lamp_alpha_mask = np.ones(
        (lamp_image_resized.shape[0], lamp_image_resized.shape[1]))
if additional_lamp_image_resized.shape[2] == 4:
    additional_lamp_alpha_mask = additional_lamp_image_resized[:, :, 3] / 255.0
else:
    additional_lamp_alpha_mask = np.ones(
        (additional_lamp_image_resized.shape[0], additional_lamp_image_resized.shape[1]))


# Merge the sofa image and the room image
sofa_merged_image = (sofa_image_resized[:, :, :3] * sofa_alpha_mask[:, :, np.newaxis]) + (
    room_image[sofa_y_coord:sofa_y_coord+sofa_image_resized.shape[0],
               sofa_x_coord:sofa_x_coord+sofa_image_resized.shape[1]] * (1 - sofa_alpha_mask[:, :, np.newaxis])
)

# Merge the painting image and the room image
painting_merged_image = (painting_image_resized[:, :, :3] * painting_alpha_mask[:, :, np.newaxis]) + (
    room_image[painting_y_coord:painting_y_coord+painting_image_resized.shape[0],
               painting_x_coord:painting_x_coord+painting_image_resized.shape[1]] * (1 - painting_alpha_mask[:, :, np.newaxis])
)

# Merge the lamp image and the room image
lamp_merged_image = (lamp_image_resized[:, :, :3] * lamp_alpha_mask[:, :, np.newaxis]) + (
    room_image[lamp_y_coord:lamp_y_coord+lamp_image_resized.shape[0],
               lamp_x_coord:lamp_x_coord+lamp_image_resized.shape[1]] * (1 - lamp_alpha_mask[:, :, np.newaxis])
)


# Merge the additional lamp image and the room image
additional_lamp_merged_image = (additional_lamp_image_resized[:, :, :3] * additional_lamp_alpha_mask[:, :, np.newaxis]) + (
    room_image[additional_lamp_y_coord:additional_lamp_y_coord+additional_lamp_image_resized.shape[0],
               additional_lamp_x_coord:additional_lamp_x_coord+additional_lamp_image_resized.shape[1]] * (1 - additional_lamp_alpha_mask[:, :, np.newaxis])
)

# Place the sofa image in the room image
room_image[sofa_y_coord:sofa_y_coord+sofa_image_resized.shape[0],
           sofa_x_coord:sofa_x_coord+sofa_image_resized.shape[1]] = sofa_merged_image

# Place the painting image in the room image
room_image[painting_y_coord:painting_y_coord+painting_image_resized.shape[0],
           painting_x_coord:painting_x_coord+painting_image_resized.shape[1]] = painting_merged_image

# Place the lamp image in the room image
room_image[lamp_y_coord:lamp_y_coord+lamp_image_resized.shape[0],
           lamp_x_coord:lamp_x_coord+lamp_image_resized.shape[1]] = lamp_merged_image


# Place the additional lamp image in the room image
room_image[additional_lamp_y_coord:additional_lamp_y_coord+additional_lamp_image_resized.shape[0],
           additional_lamp_x_coord:additional_lamp_x_coord+additional_lamp_image_resized.shape[1]] = additional_lamp_merged_image

# Display the room image with the added images
cv2.imshow('Room with Furniture', room_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image
output_path = os.path.join(output_folder, 'room_with_furniture.png')
cv2.imwrite(output_path, room_image)
print(f'Image saved at {output_path}')
