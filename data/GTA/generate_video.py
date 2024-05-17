import json, sys
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy import ndimage
from tqdm import tqdm
from multiprocessing import Pool

def get_color_map(N=256):
    """
    Return the color (R, G, B) of each label index.
    """
    
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap


# 1. Function to read video IDs from a file
def read_video_ids(file_path):
    with open(file_path) as reader:
        return [line.strip() for line in reader]

# 2. Function to load categories from a JSON file
def load_categories(file_path):
    with open(file_path) as reader:
        return json.load(reader)

# 3. Function to list files matching a pattern in a directory
def list_files(directory, pattern):
    return sorted([filename for filename in os.listdir(directory) if filename.endswith(pattern)])

# 4. Function to load an image and its mask
def load_image_and_mask(img_path, mask_path, img_file, mask_file):
    img = np.array(Image.open(os.path.join(img_path, img_file)), dtype=np.uint8)
    mask = np.load(os.path.join(mask_path, mask_file))
    return img, mask

# 5. Overlay function
def overlay(image, mask, mask_ids, colors=[255, 0, 0], cscale=1, alpha=0.4):
    colors = np.atleast_2d(colors) * cscale
    im_overlay = image.copy()
    for idx, mask_id in enumerate(mask_ids):
        if mask_id == 0: continue  # Skip background
        foreground = image * alpha + np.ones(image.shape) * (1 - alpha) * np.array(colors[idx % len(colors)])
        binary_mask = mask == mask_id
        im_overlay[binary_mask] = foreground[binary_mask]
    return im_overlay.astype(image.dtype)

# 6. Drawing tags on the image
def draw_tags_and_frame_id(image, mask, object_dict_list, font_path, frame_id):
    font = ImageFont.truetype(font_path, 30)
    draw = ImageDraw.Draw(image)
    for object_id, object_dict in enumerate(object_dict_list):
        mask_id = object_dict['object_id']
        if mask_id == 0: continue
        class_name = object_dict['category']
        tag_text = f'{class_name}-{mask_id}'
        try:
            centery, centerx = ndimage.center_of_mass(mask == mask_id)
        except:
            import pdb; pdb.set_trace()
            continue
        if np.isnan(centery) or np.isnan(centerx):
            continue
        # Draw a black rectangle
        left, top, right, bottom = draw.textbbox((0, 0), str(tag_text), font=font)
        text_width = right - left
        text_height = 2 * (bottom - top)
        draw.rectangle([centerx-5, centery-5, centerx-5+text_width, centery-5+text_height], fill=(0,0,0))
        # Write the tag text
        draw.text((centerx, centery), tag_text, font=font, fill=(255,255,255))
    # a tag to show frame id
    left, top, right, bottom = draw.textbbox((0, 0), frame_id, font=font)
    text_width = right - left
    text_height = 2 * (bottom - top)
    draw.rectangle([10, 10, 10+text_width, 10+text_height], fill=(0,0,0))
    draw.text((10, 10), frame_id, font=font, fill='white')


def draw_chart(relation_list, frame_id, font_path, num_frame):
    # Draw the relation bar chart
    last_text_end = 0.0

    bar_height = 10
    relation_height = 50
    start_height = 10  # Adjust start height as needed
    label_y_offset = 15  # Adjust as needed for text placement below the bar

    chart_width = 1280
    chart_height = relation_height * len(relation_list) + start_height

    colors = get_color_map(100)

    chart = Image.new("RGB", (chart_width, chart_height), (255, 255, 255))
    draw_bar = ImageDraw.Draw(chart)

    for idx, relation_entry in enumerate(relation_list):
        # import pdb; pdb.set_trace()
        action_start_frame, action_end_frame = relation_entry[3][0], relation_entry[3][1]
        action_label = str(relation_entry[0]) + '-' + relation_entry[1] + '-' + str(relation_entry[2])
        
        # Calculate start and end positions on the chart
        start_time = action_start_frame / num_frame * chart_width
        end_time = action_end_frame / num_frame * chart_width

        # Determine the color for the action
        action_color = tuple(colors[idx + 1])
        
        # Draw the action bar
        draw_bar.rectangle([start_time, start_height + idx * relation_height, 
                            end_time, start_height + idx * relation_height + bar_height], fill=action_color)
        
        # Calculate the width of the text
        font = ImageFont.truetype(font_path, 30)
        left, top, right, bottom = draw_bar.textbbox((0, 0), str(action_label), font=font)
        
        # Draw the label below the bar
        current_text_y = start_height + idx * relation_height + bar_height
        draw_bar.text((start_time, current_text_y), action_label, fill=action_color, font=font)
        
    indicator_x = frame_id / num_frame * chart_width
    draw_bar.line([indicator_x, 0, indicator_x, chart_height], fill=(255, 0, 0), width=5)
    
    return chart


def combine_images(mask_img, chart):
    # Determine the size of the final image
    final_width = max(mask_img.width, chart.width)
    final_height = mask_img.height + chart.height

    # Create a new blank image with the determined size
    final_image = Image.new('RGB', (final_width, final_height))

    # Paste the first image onto the new blank image
    final_image.paste(mask_img, (0, 0))

    # Paste the chart image below the first image
    final_image.paste(chart, (0, mask_img.height))
    return final_image


# 7. Creating a video from image frames
def create_video_from_frames(frame_list, output_path, frame_size, fps=5):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for frame in frame_list:
        video_writer.write(frame)
    video_writer.release()

# Example usage
font_path = '/home/jingkang001_e_ntu_edu_sg/jkyang/PSG4D/assets/OpenSans-Bold.ttf'
gta4d_json = load_categories('psg4d_gta/sailvos3d.json')

gta4d_data = gta4d_json['data']

gta4d_data = gta4d_data[:10]

for data_dict in tqdm(gta4d_data, total=len(gta4d_data)):
# def process_video(data_dict):
    video_id = data_dict['video_id']
    save_file = f"videos/{video_id}.mp4"
    # try:
    if True:
        object_dict = data_dict['object']
        object_ids = [obj['object_id'] for obj in object_dict]

        relation_list = data_dict['relations']
        meta = data_dict['meta']

        frame_list = []
        num_frames = meta['num_frames']
        for img_id in tqdm(range(num_frames), total=num_frames):
            img_name = f'{str(img_id).zfill(6)}.bmp'
            mask_name = f'{str(img_id).zfill(6)}.npy'

            img_path = f'./psg4d_gta/images/{video_id}/'
            mask_path = f'./psg4d_gta/masks/{video_id}/'

            img, mask = load_image_and_mask(img_path, mask_path, img_name, mask_name)
            colors = get_color_map(100)

            overlay_img = overlay(img, mask, object_ids, colors)
            overlay_img_pil = Image.fromarray(overlay_img)

            draw_tags_and_frame_id(overlay_img_pil, mask, object_dict, font_path, img_name.strip('.bmp'))
            chart = draw_chart(relation_list, img_id, font_path, num_frames)
            overlay_img_pil = combine_images(overlay_img_pil, chart)

            # Convert PIL image back to array for video creation
            overlay_img_np = np.array(overlay_img_pil)
            overlay_img_np = cv2.cvtColor(overlay_img_np, cv2.COLOR_RGB2BGR)
            
            frame_list.append(overlay_img_np)

        create_video_from_frames(frame_list, save_file, (overlay_img_np.shape[1], overlay_img_np.shape[0]))

    # except Exception as e:
    #     print(f"{data_dict['video_id']} Error: {e}", flush=True)


# job_id = int(sys.argv[1])
# gta4d_data = gta4d_data[40*(job_id-1):40*job_id]
# with Pool(processes=40) as pool:
#     data_list = list(tqdm(pool.imap(process_video, gta4d_data), total=len(gta4d_data)))

# python generate_video.py
# with Pool(processes=40) as pool:
#     data_list = list(tqdm(pool.imap(process_video, gta4d_data), total=len(gta4d_data)))
