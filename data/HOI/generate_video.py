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
    mask = np.load(os.path.join(mask_path, mask_file))['array']
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
def draw_tags_and_frame_id(image, mask, mask_ids, object_ids, categories, font_path, frame_id):
    font = ImageFont.truetype(font_path, 30)
    draw = ImageDraw.Draw(image)
    for mask_id, object_id in zip(mask_ids, object_ids):
        if mask_id == 0: continue
        class_id = mask_id % 1000
        class_name = categories[str(class_id)]
        tag_text = f'{class_name}-{object_id}'
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
    draw.rectangle([10, 1000, 10+text_width, 1000+text_height], fill=(0,0,0))
    draw.text((10, 1000), frame_id, font=font, fill='white')


def draw_chart(relation_list, frame_id, font_path):
    # Draw the relation bar chart
    last_text_end = 0.0

    bar_height = 10
    start_height = 10  # Adjust start height as needed
    label_y_offset = 15  # Adjust as needed for text placement below the bar
    num_frame = 300

    current_text_y = start_height + bar_height + label_y_offset

    chart_width = 1920
    chart_height = 200

    colors = get_color_map(100)

    chart = Image.new("RGB", (chart_width, chart_height), (255, 255, 255))
    draw_bar = ImageDraw.Draw(chart)

    for idx, relation_entry in enumerate(relation_list):
        action_start_frame, action_end_frame = relation_entry[3], relation_entry[4]
        action_label = str(relation_entry[0]) + '-' + relation_entry[2] + '-' + str(relation_entry[1])
        
        # Calculate start and end positions on the chart
        start_time = action_start_frame / num_frame * chart_width
        end_time = action_end_frame / num_frame * chart_width

        # Determine the color for the action
        action_color = tuple(colors[idx + 1])
        
        # Draw the action bar
        draw_bar.rectangle([start_time, start_height, end_time, start_height + bar_height], fill=action_color)
        
        # Calculate the width of the text
        font = ImageFont.truetype(font_path, 30)
        # text_width, text_height = draw_bar.textsize(action_label, font=font)

        left, top, right, bottom = draw_bar.textbbox((0, 0), str(action_label), font=font)
        text_width = right - left
        text_height = bottom - top

        text_x = start_time
        
        if text_x > last_text_end:
            last_text_end = 0

        # Check if the text overlaps the previous text, if so, move it to the next row
        if text_x < last_text_end:
            current_text_y += text_height  # Move to the next row
        else:
            current_text_y = label_y_offset
            
        # Draw the label below the bar
        draw_bar.text((text_x, current_text_y), action_label, fill=action_color, font=font)

        # Update the last text end position
        last_text_end = text_x + text_width
        
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
def create_video_from_frames(frame_list, output_path, frame_size, fps=60):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for frame in frame_list:
        video_writer.write(frame)
    video_writer.release()

# Example usage
font_path = '/home/jingkang001_e_ntu_edu_sg/jkyang/PSG4D/assets/OpenSans-Bold.ttf'
categories = load_categories('categories.json')
hoi4d_json = load_categories('psg4d_hoi.json')
hoi4d_id2v = load_categories('hoi4d_id.json')
hoi4d_v2id = {v: k for k, v in hoi4d_id2v.items()}

hoi4d_data = hoi4d_json['data']

hoi4d_data = hoi4d_data[:10]

for data_dict in tqdm(hoi4d_data, total=len(hoi4d_data)):
# def process_video(data_dict):
    video_id = data_dict['video_id']
    save_file = f"videos/{hoi4d_v2id[video_id]}.mp4"
    try:
        object_dict = data_dict['object']
        mask_ids = [obj['mask_id'] for obj in object_dict]
        object_ids = [obj['object_id'] for obj in object_dict]

        relation_list = data_dict['relations']

        frame_list = []
        for img_id in tqdm(range(300), total=300):
            img_name = f'{str(img_id).zfill(5)}.jpg'
            mask_name = f'{str(img_id).zfill(5)}.npz'

            img_path = f'./HOI4D_release/{video_id}/align_rgb'
            mask_path = f'./HOI4D_mask/{video_id}/mask'

            img, mask = load_image_and_mask(img_path, mask_path, img_name, mask_name)
            colors = get_color_map(100)

            overlay_img = overlay(img, mask.T, mask_ids, colors)
            overlay_img_pil = Image.fromarray(overlay_img)

            draw_tags_and_frame_id(overlay_img_pil, mask.T, mask_ids, object_ids, categories, font_path, img_name.strip('.jpg'))
            chart = draw_chart(relation_list, img_id, font_path)
            overlay_img_pil = combine_images(overlay_img_pil, chart)

            # Convert PIL image back to array for video creation
            overlay_img_np = np.array(overlay_img_pil)
            overlay_img_np = cv2.cvtColor(overlay_img_np, cv2.COLOR_RGB2BGR)
            
            frame_list.append(overlay_img_np)

        create_video_from_frames(frame_list, save_file, (overlay_img_np.shape[1], overlay_img_np.shape[0]))

    except Exception as e:
        print(f"{data_dict['video_id']} Error: {e}", flush=True)


# job_id = int(sys.argv[1])
# hoi4d_data = hoi4d_data[40*(job_id-1):40*job_id]
# with Pool(processes=40) as pool:
#     data_list = list(tqdm(pool.imap(process_video, hoi4d_data), total=len(hoi4d_data)))

# python generate_video.py
# with Pool(processes=40) as pool:
#     data_list = list(tqdm(pool.imap(process_video, hoi4d_data), total=len(hoi4d_data)))
