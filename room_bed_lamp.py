import cv2
import os
import numpy as np

# Load the original room, bed, and painting images
room_image_path = './images/room/room8.png'
bed_image_path = './images/bed/bed15.png'
painting_image_path = './images/painting/painting.png'
lamp_image_path = './images/lamp/lamp.png'
output_folder = './output/'

room_image = cv2.imread(room_image_path, cv2.IMREAD_COLOR)
bed_image = cv2.imread(bed_image_path, cv2.IMREAD_UNCHANGED)
painting_image = cv2.imread(painting_image_path, cv2.IMREAD_UNCHANGED)
lamp_image = cv2.imread(lamp_image_path, cv2.IMREAD_UNCHANGED)

# Define the scale factor for resizing the bed image
# Adjust the scale factor as desired (0.8 reduces the size by 20%)
bed_scale_factor = 0.8

# Resize the bed image with the scale factor
bed_image_resized = cv2.resize(
    bed_image, None, fx=bed_scale_factor, fy=bed_scale_factor)

# Increase the height of the bed image slightly
bed_new_height = int(bed_image_resized.shape[0] * 1.2)  # Increase by 20%
bed_image_resized = cv2.resize(
    bed_image_resized, (bed_image_resized.shape[1], bed_new_height))

# Find the dimensions of the resized bed image
bed_height, bed_width = bed_image_resized.shape[:2]

# Define the coordinates to place the bed image in the room image
bed_x_coord = 115  # Update with the desired x-coordinate
bed_y_coord = 312  # Update with the desired y-coordinate

# Calculate the end coordinates for placing the bed image
bed_x_end = bed_x_coord + bed_width
bed_y_end = bed_y_coord + bed_height

# Resize the room region to match the dimensions of the bed image
bed_room_region_resized = cv2.resize(
    room_image[bed_y_coord:bed_y_end, bed_x_coord:bed_x_end], (bed_width, bed_height))

# Create a mask for the bed image
if bed_image_resized.shape[2] == 4:
    bed_alpha_mask = bed_image_resized[:, :, 3] / 255.0
else:
    bed_alpha_mask = np.ones((bed_height, bed_width))

# Create an inverted mask for the room region
bed_room_mask = 1.0 - bed_alpha_mask

# Merge the bed image and the room region
bed_merged_image = (bed_image_resized[:, :, :3] * bed_alpha_mask[:, :, np.newaxis]) + (
    bed_room_region_resized * bed_room_mask[:, :, np.newaxis]
)

# Resize the merged image to match the size of the room region
bed_merged_image_resized = cv2.resize(
    bed_merged_image, (bed_x_end - bed_x_coord, bed_y_end - bed_y_coord))

# Update the corresponding region in the original room image
room_image[bed_y_coord:bed_y_end,
           bed_x_coord:bed_x_end] = bed_merged_image_resized

# Define the scale factor for resizing the painting image
# Adjust the scale factor as desired (0.6 reduces the size by 40%)
painting_scale_factor = 0.5

# Resize the painting image with the scale factor
painting_image_resized = cv2.resize(
    painting_image, None, fx=painting_scale_factor, fy=painting_scale_factor)

# Find the dimensions of the resized painting image
painting_height, painting_width = painting_image_resized.shape[:2]

# Increase the width of the painting image slightly
painting_new_width = int(painting_width * 1.2)  # Increase by 20%
painting_image_resized = cv2.resize(
    painting_image_resized, (painting_new_width, painting_height))

# Define the coordinates to place the painting image in the room image
painting_x_coord = 180  # Update with the desired x-coordinate
painting_y_coord = 190  # Update with the desired y-coordinate

# Resize the room region to match the dimensions of the painting image
painting_room_region_resized = cv2.resize(
    room_image[painting_y_coord: painting_y_coord + painting_height,
               painting_x_coord: painting_x_coord + painting_new_width],
    (painting_new_width, painting_height),
)

# Create a mask for the painting image
if painting_image_resized.shape[2] == 4:
    painting_alpha_mask = painting_image_resized[:, :, 3] / 255.0
else:
    painting_alpha_mask = np.ones((painting_height, painting_new_width))

# Create an inverted mask for the room region
painting_room_mask = 1.0 - painting_alpha_mask

# Merge the painting image and the room region
painting_merged_image = (painting_image_resized[:, :, :3] * painting_alpha_mask[:, :, np.newaxis]) + (
    painting_room_region_resized * painting_room_mask[:, :, np.newaxis]
)

# Resize the merged image to match the size of the room region
painting_merged_image_resized = cv2.resize(
    painting_merged_image, (painting_x_coord + painting_new_width - painting_x_coord,
                            painting_y_coord + painting_height - painting_y_coord)
)

# Update the corresponding region in the original room image
room_image[painting_y_coord: painting_y_coord + painting_height,
           painting_x_coord: painting_x_coord + painting_new_width] = painting_merged_image_resized

# Define the scale factor for resizing the lamp image
# Adjust the scale factor as desired (0.8 reduces the size by 20%)
lamp_scale_factor = 0.3

# Resize the lamp image with the scale factor
lamp_image_resized = cv2.resize(
    lamp_image, None, fx=lamp_scale_factor, fy=lamp_scale_factor)

# Find the dimensions of the resized lamp image
lamp_height, lamp_width = lamp_image_resized.shape[:2]

# Define the coordinates to place the lamp image in the room image
lamp_x_coord = 399  # Update with the desired x-coordinate
lamp_y_coord = 332  # Update with the desired y-coordinate

# Increase the width of the lamp image slightly
lamp_new_width = int(lamp_width * 1.3)  # Increase by 20%
lamp_image_resized = cv2.resize(
    lamp_image_resized, (lamp_new_width, lamp_height))

# Resize the room region to match the dimensions of the lamp image
lamp_room_region_resized = cv2.resize(
    room_image[lamp_y_coord: lamp_y_coord + lamp_height,
               lamp_x_coord: lamp_x_coord + lamp_new_width],
    (lamp_new_width, lamp_height),
)

# Create a mask for the lamp image
if lamp_image_resized.shape[2] == 4:
    lamp_alpha_mask = lamp_image_resized[:, :, 3] / 255.0
else:
    lamp_alpha_mask = np.ones((lamp_height, lamp_new_width))

# Create an inverted mask for the room region
lamp_room_mask = 1.0 - lamp_alpha_mask

# Merge the lamp image and the room region
lamp_merged_image = (lamp_image_resized[:, :, :3] * lamp_alpha_mask[:, :, np.newaxis]) + (
    lamp_room_region_resized * lamp_room_mask[:, :, np.newaxis]
)

# Resize the merged image to match the size of the room region
lamp_merged_image_resized = cv2.resize(
    lamp_merged_image, (lamp_x_coord + lamp_new_width - lamp_x_coord,
                        lamp_y_coord + lamp_height - lamp_y_coord)
)

# Update the corresponding region in the original room image
room_image[lamp_y_coord: lamp_y_coord + lamp_height,
           lamp_x_coord: lamp_x_coord + lamp_new_width] = lamp_merged_image_resized

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the modified room image
output_path = os.path.join(output_folder, 'modified_room.png')
cv2.imwrite(output_path, room_image)

# Display the modified room image
cv2.imshow('Modified Room Image', room_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
