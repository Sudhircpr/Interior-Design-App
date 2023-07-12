import cv2
import os
import numpy as np

# Load the original room, bed, sofa, painting, lamp, chair, and table images
room_image_path = './images/room/room.jpg'  # Path to the blank room image
bed_image_path = './images/bed/bed.png'  # Path to the bed image
sofa_image_path = './images/sofa/sofa.png'  # Path to the sofa image
painting_image_path = './images/painting/painting1.png'  # Path to the painting image
lamp_image_path = './images/lamp/lamp1.png'  # Path to the lamp image
chair_image_path = './images/chair/chair1.png'  # Path to the chair image
table_image_path = './images/table/table1.png'  # Path to the table image
# Path to the additional lamp image
additional_lamp_image_path = './images/lamp/lamp2.png'
output_folder = './output/'

room_image = cv2.imread(room_image_path, cv2.IMREAD_COLOR)
bed_image = cv2.imread(bed_image_path, cv2.IMREAD_UNCHANGED)
sofa_image = cv2.imread(sofa_image_path, cv2.IMREAD_UNCHANGED)
painting_image = cv2.imread(painting_image_path, cv2.IMREAD_UNCHANGED)
lamp_image = cv2.imread(lamp_image_path, cv2.IMREAD_UNCHANGED)
chair_image = cv2.imread(chair_image_path, cv2.IMREAD_UNCHANGED)
table_image = cv2.imread(table_image_path, cv2.IMREAD_UNCHANGED)
additional_lamp_image = cv2.imread(
    additional_lamp_image_path, cv2.IMREAD_UNCHANGED)

# Define the coordinates to place the bed, sofa, painting, lamp, chair, table, and additional lamp images in the room image
bed_x_coord = 400  # Update with the desired x-coordinate for the bed
bed_y_coord = 410  # Update with the desired y-coordinate for the bed
sofa_x_coord = 110  # Update with the desired x-coordinate for the sofa
sofa_y_coord = 510  # Update with the desired y-coordinate for the sofa
painting_x_coord = 360  # Update with the desired x-coordinate for the painting
painting_y_coord = 200  # Update with the desired y-coordinate for the painting
lamp_x_coord = 600  # Update with the desired x-coordinate for the lamp
lamp_y_coord = 140  # Update with the desired y-coordinate for the lamp
chair_x_coord = 50  # Update with the desired x-coordinate for the chair
chair_y_coord = 620  # Update with the desired y-coordinate for the chair
table_x_coord = 120  # Update with the desired x-coordinate for the table
table_y_coord = 620  # Update with the desired y-coordinate for the table
# Update with the desired x-coordinate for the additional lamp
additional_lamp_x_coord = 115
# Update with the desired y-coordinate for the additional lamp
additional_lamp_y_coord = 140

# Define the scale factors for resizing the bed, sofa, painting, lamp, chair, table, and additional lamp images
# Adjust the scale factors as desired
bed_scale_factor = 0.5
sofa_scale_factor = 0.3
painting_scale_factor = 0.2
lamp_scale_factor = 0.3
chair_scale_factor = 0.1
table_scale_factor = 0.3
additional_lamp_scale_factor = 0.3

# Resize the bed, sofa, painting, lamp, chair, table, and additional lamp images with the scale factors
bed_image_resized = cv2.resize(
    bed_image, None, fx=bed_scale_factor, fy=bed_scale_factor)
sofa_image_resized = cv2.resize(
    sofa_image, None, fx=sofa_scale_factor, fy=sofa_scale_factor)
painting_image_resized = cv2.resize(
    painting_image, None, fx=painting_scale_factor, fy=painting_scale_factor)
lamp_image_resized = cv2.resize(
    lamp_image, None, fx=lamp_scale_factor, fy=lamp_scale_factor)
chair_image_resized = cv2.resize(
    chair_image, None, fx=chair_scale_factor, fy=chair_scale_factor)
table_image_resized = cv2.resize(
    table_image, None, fx=table_scale_factor, fy=table_scale_factor)
additional_lamp_image_resized = cv2.resize(
    additional_lamp_image, None, fx=additional_lamp_scale_factor, fy=additional_lamp_scale_factor)

# Create masks for the bed, sofa, painting, lamp, chair, table, and additional lamp images
if bed_image_resized.shape[2] == 4:
    bed_alpha_mask = bed_image_resized[:, :, 3] / 255.0
else:
    bed_alpha_mask = np.ones(
        (bed_image_resized.shape[0], bed_image_resized.shape[1]))

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

if chair_image_resized.shape[2] == 4:
    chair_alpha_mask = chair_image_resized[:, :, 3] / 255.0
else:
    chair_alpha_mask = np.ones(
        (chair_image_resized.shape[0], chair_image_resized.shape[1]))

if table_image_resized.shape[2] == 4:
    table_alpha_mask = table_image_resized[:, :, 3] / 255.0
else:
    table_alpha_mask = np.ones(
        (table_image_resized.shape[0], table_image_resized.shape[1]))

if additional_lamp_image_resized.shape[2] == 4:
    additional_lamp_alpha_mask = additional_lamp_image_resized[:, :, 3] / 255.0
else:
    additional_lamp_alpha_mask = np.ones(
        (additional_lamp_image_resized.shape[0], additional_lamp_image_resized.shape[1]))

# Merge the bed image and the room image
bed_merged_image = (bed_image_resized[:, :, :3] * bed_alpha_mask[:, :, np.newaxis]) + (
    room_image[bed_y_coord:bed_y_coord+bed_image_resized.shape[0],
               bed_x_coord:bed_x_coord+bed_image_resized.shape[1]] * (1 - bed_alpha_mask[:, :, np.newaxis])
)

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

# Merge the chair image and the room image
chair_merged_image = (chair_image_resized[:, :, :3] * chair_alpha_mask[:, :, np.newaxis]) + (
    room_image[chair_y_coord:chair_y_coord+chair_image_resized.shape[0],
               chair_x_coord:chair_x_coord+chair_image_resized.shape[1]] * (1 - chair_alpha_mask[:, :, np.newaxis])
)

# Merge the table image and the room image
table_merged_image = (table_image_resized[:, :, :3] * table_alpha_mask[:, :, np.newaxis]) + (
    room_image[table_y_coord:table_y_coord+table_image_resized.shape[0],
               table_x_coord:table_x_coord+table_image_resized.shape[1]] * (1 - table_alpha_mask[:, :, np.newaxis])
)

# Merge the additional lamp image and the room image
additional_lamp_merged_image = (additional_lamp_image_resized[:, :, :3] * additional_lamp_alpha_mask[:, :, np.newaxis]) + (
    room_image[additional_lamp_y_coord:additional_lamp_y_coord+additional_lamp_image_resized.shape[0],
               additional_lamp_x_coord:additional_lamp_x_coord+additional_lamp_image_resized.shape[1]] * (1 - additional_lamp_alpha_mask[:, :, np.newaxis])
)

# Place the bed image in the room image
room_image[bed_y_coord:bed_y_coord+bed_image_resized.shape[0],
           bed_x_coord:bed_x_coord+bed_image_resized.shape[1]] = bed_merged_image

# Place the sofa image in the room image
room_image[sofa_y_coord:sofa_y_coord+sofa_image_resized.shape[0],
           sofa_x_coord:sofa_x_coord+sofa_image_resized.shape[1]] = sofa_merged_image

# Place the painting image in the room image
room_image[painting_y_coord:painting_y_coord+painting_image_resized.shape[0],
           painting_x_coord:painting_x_coord+painting_image_resized.shape[1]] = painting_merged_image

# Place the lamp image in the room image
room_image[lamp_y_coord:lamp_y_coord+lamp_image_resized.shape[0],
           lamp_x_coord:lamp_x_coord+lamp_image_resized.shape[1]] = lamp_merged_image

# Place the chair image in the room image
room_image[chair_y_coord:chair_y_coord+chair_image_resized.shape[0],
           chair_x_coord:chair_x_coord+chair_image_resized.shape[1]] = chair_merged_image

# Place the table image in the room image
room_image[table_y_coord:table_y_coord+table_image_resized.shape[0],
           table_x_coord:table_x_coord+table_image_resized.shape[1]] = table_merged_image

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
