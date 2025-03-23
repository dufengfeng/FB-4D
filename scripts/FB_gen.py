import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
import numpy
import os
import sys
import shutil
import numpy
import torch
import rembg
import threading
import urllib.request
from PIL import Image
import streamlit as st
import json
import huggingface_hub
from app import SAMAPI

best_view = []
all_clip_score = []
clip_score_ref = []
ref_list = []
ref_azimuth_list = []
camera =[
    {"elevation": 20, "azimuth": 30},
    {"elevation": -10, "azimuth": 90},
    {"elevation": 20, "azimuth": 150},
    {"elevation": -10, "azimuth": 210},
    {"elevation": 20, "azimuth": 270},
    {"elevation": -10, "azimuth": 330}
]
weight_list_ALL = []
# Load the CLIP model
model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID)
preprocess = CLIPImageProcessor.from_pretrained(model_ID)

def filter_camera_by_indices(camera, index_list):
    return [camera[i] for i in index_list]

def create_directory_if_not_exists(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def segment_img(img: Image):
    output = rembg.remove(img)
    mask = numpy.array(output)[:, :, 3] > 0
    sam_mask = SAMAPI.segment_api(numpy.array(img)[:, :, :3], mask)
    segmented_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    segmented_img.paste(img, mask=Image.fromarray(sam_mask))
    return segmented_img


def segment_6imgs(zero123pp_imgs):
    imgs = [zero123pp_imgs.crop([0, 0, 320, 320]),
            zero123pp_imgs.crop([320, 0, 640, 320]),
            zero123pp_imgs.crop([0, 320, 320, 640]),
            zero123pp_imgs.crop([320, 320, 640, 640]),
            zero123pp_imgs.crop([0, 640, 320, 960]),
            zero123pp_imgs.crop([320, 640, 640, 960])]
    segmented_imgs = []
    import numpy as np
    for i, img in enumerate(imgs):
        output = rembg.remove(img)
        mask = numpy.array(output)[:, :, 3]
        mask = SAMAPI.segment_api(numpy.array(img)[:, :, :3], mask)
        data = numpy.array(img)[:,:,:3]
        data2 = numpy.ones([320,320,4])
        data2[:,:,:3] = data
        for i in np.arange(data2.shape[0]):
                for j in np.arange(data2.shape[1]):
                        if mask[i,j]==1:
                                data2[i,j,3]=255
        segmented_imgs.append(data2)

        #torch.manual_seed(42)
    return segmented_imgs

def process_img(path,destination,pipeline, is_first, info_strength, warm = False, weight_b1 = 0, weight_b2 = 0, weight_b3 = 0):
    # Download an example image.
        print('processing:',path)
        if path=="":
            cond = None
        else:
            cond = Image.open(path)
        # Run the pipeline!
        result = pipeline(cond, num_inference_steps=75,is_first = is_first,info_strength = info_strength, weight_b1 = weight_b1, weight_b2 = weight_b2, weight_b3 = weight_b3).images[0]
        
        if warm == False:
            result.save("output.png")
            result=segment_6imgs(result)
            file_len = len(os.listdir(destination))
            print('saving:',os.path.join(destination,f'{file_len:02d}~{(file_len+5):02d}.png'),'in',destination)
            for i in numpy.arange(6):
                Image.fromarray(numpy.uint8(result[i])).save(os.path.join(destination,'{:02d}.png'.format(i+file_len)))


def copy_and_rename_image(source_dir, dest_dir, img_name_pattern):
    """Copy and rename images from source to destination directory."""
    create_directory_if_not_exists(dest_dir)
    dest_len = len(os.listdir(dest_dir))
    file_name = f'{dest_len:02d}_rgba.png'
    source_file = os.path.join(source_dir,img_name_pattern)
    if os.path.exists(source_file):
        destination_file = os.path.join(dest_dir, file_name)
        shutil.copy(source_file, destination_file)


def add_camera(index):
    elevation_val = camera[index]["elevation"]
    azimuth_val = camera[index]["azimuth"]
    for i in range(6):
        new_elevation = elevation_val + (20 if i % 2 == 0 else -10)
        new_azimuth = (azimuth_val + 30 + i * 60) % 360
        camera.append({"elevation": new_elevation, "azimuth": new_azimuth})
     

def clip_contrast(source_dir, view_num):
    clip = []
    for i in range(0,32):
        # find_the file with the pattern
        image_1 = Image.open(os.path.join(source_dir,"ref/",f'{i:02d}_rgba.png'))
        image_2 = Image.open(os.path.join(source_dir,"zero123/",f'{i:02d}_rgba/',f"{view_num:02d}.png"))
        image_2 = image_2.resize((image_1.size[0], image_1.size[1]))
        gr_bg = Image.new("RGBA", image_1.size, (128, 128, 128, 255))
        image_1 = Image.alpha_composite(gr_bg, image_1)
        image_2 = Image.alpha_composite(gr_bg, image_2)
        image_1 = image_1.convert("RGB")
        image_2 = image_2.convert("RGB")
        image_1 = preprocess(image_1, return_tensors="pt")["pixel_values"]
        image_2 = preprocess(image_2, return_tensors="pt")["pixel_values"]

        with torch.no_grad():
            embedding_pred = model.get_image_features(image_1)
            embedding_gt = model.get_image_features(image_2)

            # Calculate the cosine similarity between the embeddings
            clip_score = torch.nn.functional.cosine_similarity(embedding_pred, embedding_gt)
            clip.append(clip_score)
    
    average_clip_score = torch.mean(torch.stack(clip)).item()
    # print(f"the score of {view_num} is {average_clip_score}")
    return average_clip_score

def clip_contrast_view(source_dir, view_num_1, view_num_2):
    clip = []
    for i in range(0,32):
        # find_the file with the pattern
        image_1 = Image.open(os.path.join(source_dir,"zero123/",f'{i:02d}_rgba/',f"{view_num_1:02d}.png"))
        image_2 = Image.open(os.path.join(source_dir,"zero123/",f'{i:02d}_rgba/',f"{view_num_2:02d}.png"))
        image_2 = image_2.resize((image_1.size[0], image_1.size[1]))
        gr_bg = Image.new("RGBA", image_1.size, (128, 128, 128, 255))
        image_1 = Image.alpha_composite(gr_bg, image_1)
        image_2 = Image.alpha_composite(gr_bg, image_2)
        image_1 = image_1.convert("RGB")
        image_2 = image_2.convert("RGB")
        image_1 = preprocess(image_1, return_tensors="pt")["pixel_values"]
        image_2 = preprocess(image_2, return_tensors="pt")["pixel_values"]

        with torch.no_grad():
            embedding_pred = model.get_image_features(image_1)
            embedding_gt = model.get_image_features(image_2)

            # Calculate the cosine similarity between the embeddings
            clip_score = torch.nn.functional.cosine_similarity(embedding_pred, embedding_gt)
            clip.append(clip_score)
    
    average_clip_score = torch.mean(torch.stack(clip)).item()
    # print(f"the score of {view_num} is {average_clip_score}")
    return average_clip_score


def return_idx(source_dir, generation_idx):

    if generation_idx == 0:
        for i in range(generation_idx * 6, (generation_idx + 1) * 6):
            all_clip_score.append(clip_contrast(source_dir, i))
    else:
        for now_index in range(generation_idx * 6, (generation_idx + 1) * 6):
            clip_score_now = 0
            weight_list = []
            for i in range(0, len(ref_azimuth_list)):
                diff1 = abs(ref_azimuth_list[i] - camera[now_index]["azimuth"])
                diff2 = 360 - abs(ref_azimuth_list[i] - camera[now_index]["azimuth"])
                diff = min(diff1, diff2)
                weight = 180 - diff
                weight_list.append(weight)
            # normalize weight_list
            weight_list = [x / sum(weight_list) for x in weight_list]

            # cal corresponding clip_score
            for i in range(0, len(ref_azimuth_list)):
                clip_score_each = clip_contrast_view(source_dir, now_index, ref_list[i]) * weight_list[i] * 0.5
                clip_score_now += clip_score_each
            clip_score_now += clip_contrast(source_dir, now_index) * 0.5
            all_clip_score.append(clip_score_now)
    
    if best_view == []:
        best_view_now = max(all_clip_score)
        best_view.append(best_view_now)
        best_view_index = all_clip_score.index(best_view_now)
        # add_camera(best_view_index)
        ref_list.append(best_view_index)
        ref_azimuth_list.append(camera[best_view_index]["azimuth"])
        return best_view_index
    else:
        # need to be more close to the backangle
        max_angle = abs(ref_azimuth_list[-1] - 180)

        all_images = []
        all_images_clip_score = []
        candidate = []
        candidate_clip_score = []

        for i in range(0, len(camera)):
            if all_clip_score[i] not in best_view:
                if abs(camera[i]["elevation"])<=30:
                    all_images.append(i)
                    all_images_clip_score.append(all_clip_score[i])
                    if abs(camera[i]["azimuth"]-180) <= max_angle:
                        candidate.append(i)
                        candidate_clip_score.append(all_clip_score[i])
        
        if candidate_clip_score == []:
            max_clip_score = max(all_images_clip_score)
            max_clip_score_index = all_images_clip_score.index(max_clip_score)
            initial_index = all_images[max_clip_score_index]
            best_view.append(max_clip_score)
            best_view_index = initial_index
            
            ref_list.append(best_view_index)
            ref_azimuth_list.append(camera[best_view_index]["azimuth"])
            return best_view_index
        else:
            max_clip_score = max(candidate_clip_score)
            max_clip_score_index = candidate_clip_score.index(max_clip_score)
            initial_index = candidate[max_clip_score_index]
            best_view.append(max_clip_score)
            best_view_index = initial_index
            
            ref_list.append(best_view_index)
            ref_azimuth_list.append(camera[best_view_index]["azimuth"])
            return best_view_index
    
def justify_data(source_dir, number):
    for i in range(0, number):
        clip_score_ref.append(clip_contrast(source_dir, i))
        
    last_index = []

    initial_score = {
                    camera[0]["azimuth"]:all_clip_score[0],
                    camera[1]["azimuth"]:all_clip_score[1],
                    camera[2]["azimuth"]:all_clip_score[2],
                    camera[3]["azimuth"]:all_clip_score[3],
                    camera[4]["azimuth"]:all_clip_score[4],
                    camera[5]["azimuth"]:all_clip_score[5]
                    }
    low_score = min(initial_score.values())
    for i in range(0, 12):
        if i%2 == 0:
            # get the neighbour score
            # left_azimuth = (i * 30 - 30 + 360) % 360
            # right_azimuth = (i * 30 + 30) % 360
            # left_score = initial_score[left_azimuth]
            # right_score = initial_score[right_azimuth]
            now_azimuth = i * 30
            fit_index = [index for index, cam in enumerate(camera) if cam["azimuth"] == now_azimuth]
            max_fit_score = max(clip_score_ref[temp] for temp in fit_index)
            max_fit_index = clip_score_ref.index(max_fit_score)
            if max_fit_score >= 0.8:
                if max_fit_score >= low_score - 0.01:
                    for others_index in fit_index:
                        if abs(clip_score_ref[others_index] - max_fit_score) < 0.01:
                            last_index.append(others_index)
            else:
                neighbour_azimuth = (i * 30 + 30) % 360
                neighbour_index = [index for index, cam in enumerate(camera) if cam["azimuth"] == neighbour_azimuth]
                max_neighbour_score = max(clip_score_ref[temp] for temp in neighbour_index)
                neighbour_fit_index = clip_score_ref.index(max_neighbour_score)
                
                if max_neighbour_score <= 0.8:
                    if max_neighbour_score >= max_fit_score:
                        last_index.append(neighbour_fit_index)
                    else:
                        last_index.append(max_fit_index)
        else:
            # next picking
            now_azimuth = i * 30
            fit_index = [index for index, cam in enumerate(camera) if cam["azimuth"] == now_azimuth]
            max_fit_score = max(clip_score_ref[temp] for temp in fit_index)
            max_fit_index = clip_score_ref.index(max_fit_score)
            if max_fit_score >= 0.8:
                for others_index in fit_index:
                    if abs(clip_score_ref[others_index] - max_fit_score) < 0.01:
                        last_index.append(others_index)
            else:
                neighbour_azimuth = (i * 30 + 30 + 360) % 360
                neighbour_index = [index for index, cam in enumerate(camera) if cam["azimuth"] == neighbour_azimuth]
                max_neighbour_score = max(clip_score_ref[temp] for temp in neighbour_index)
                neighbour_fit_index = clip_score_ref.index(max_neighbour_score)

                if max_neighbour_score <= 0.8:
                    if max_neighbour_score >= max_fit_score:
                        last_index.append(neighbour_fit_index)
                    else:
                        last_index.append(max_fit_index)

    return last_index

# featurebank correspondence info(strengthen temporal consistency)
# progressively generating featurebank(clip score)

def image_iter(id, info_strength, weight_b1, weight_b2, weight_b3):
    # preprocess
    if id == 1:
        pth = "ref"
    else:
        pth = f"ref{id}"

    # warmup
    cnt = 0
    is_first = True
    # featurebank initialization
    for file in sorted(os.listdir(directory + pth)):
        if(cnt >= 1):
            is_first = False
        if(cnt >= 5):
            break
        source_dir1 = os.path.join(directory, pth)
        dest_dir = os.path.join(directory, 'zero123')
        if file.endswith('.png'):
            filename = os.path.splitext(os.path.basename(file))[0]  
            img_path = os.path.join(source_dir1, file)
            destination = os.path.join(dest_dir, filename)
            process_img(img_path, destination, pipeline, is_first, info_strength = info_strength, warm = True, weight_b1 = weight_b1, weight_b2 = weight_b2, weight_b3 = weight_b3)
        cnt += 1
    cnt = 0

    # begin pipeline
    is_first = False
    for file in sorted(os.listdir(directory + pth)):
        source_dir1 = os.path.join(directory, pth)
        dest_dir = os.path.join(directory, 'zero123')
        if file.endswith('.png'):
            filename = os.path.splitext(os.path.basename(file))[0]
            file_number = filename.split('_')[0]        
            destination = os.path.join(dest_dir, filename)
            create_directory_if_not_exists(destination)
            img_path = os.path.join(source_dir1, file)
            process_img(img_path, destination, pipeline, is_first, info_strength = info_strength, weight_b1 = weight_b1, weight_b2 = weight_b2, weight_b3 = weight_b3)
        cnt += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to process")
    # DiffusionPipeline.from_pretrained cannot received relative path for custom pipeline
    parser.add_argument("--pipeline_path", required=True, help="path of pipeline code")
    parser.add_argument("--number",required=False, help="number of multi-images to generate", type=int, default=18)
    args, extras = parser.parse_known_args()

    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", custom_pipeline=args.pipeline_path,
        torch_dtype=torch.float32
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    pipeline.to('cuda:0')
    directory = args.path + '/'

    # Prepare directories
    create_directory_if_not_exists(directory + 'ref')
    os.system(f"cp -r {directory+'*.png'} {directory+'ref/'}")
    if(args.number >= 12):
        create_directory_if_not_exists(directory + 'ref2')
    if(args.number >= 18):
        create_directory_if_not_exists(directory + 'ref3')


    if (args.number >= 6):

        image_iter(1, info_strength=1, weight_b1 = 0, weight_b2 = 0, weight_b3 = 0)
        next_id = return_idx(args.path, 0)
        for i in range(0,32):
            filename = f"{i:02d}_rgba"
            copy_and_rename_image(args.path + '/zero123/' + filename, directory + 'ref2', f"{next_id:02d}.png")
    
    if(ref_azimuth_list[-1]!=180):
        if(args.number >= 12):
            add_camera(next_id)
            diff_b1 = min(abs(ref_azimuth_list[-1]), 360 - abs(ref_azimuth_list[-1]))
            weight_b1 = (180 - diff_b1) / 360
            weight_b2 = 0
            weight_b3 = 0
            weight_list_ALL.append(weight_b1)
            weight_list_ALL.append(weight_b2)
            weight_list_ALL.append(weight_b3)

            image_iter(2, info_strength=2, weight_b1 = weight_b1, weight_b2 = weight_b2, weight_b3 = weight_b3)
            next_id = return_idx(args.path, 1)
            for i in range(0,32):
                filename = f"{i:02d}_rgba"
                copy_and_rename_image(args.path + '/zero123/' + filename, directory + 'ref3', f"{next_id:02d}.png")
        
    
    if(ref_azimuth_list[-1]!=180):
        if(args.number >= 18):
            add_camera(next_id)

            diff_b1 = min(abs(ref_azimuth_list[-1]), 360 - abs(ref_azimuth_list[-1]))
            diff_b2 = min(abs(ref_azimuth_list[-1] - ref_azimuth_list[-2]), 360 - abs(ref_azimuth_list[-1] - ref_azimuth_list[-2]))

            weight_b1 = (180 - diff_b1) / 720
            weight_b2 = (180 - diff_b2) / 720
            weight_b3 = 0

            weight_list_ALL.append(weight_b1)
            weight_list_ALL.append(weight_b2)
            weight_list_ALL.append(weight_b3)

            image_iter(3, info_strength=3, weight_b1 = weight_b1, weight_b2 = weight_b2, weight_b3 = weight_b3)
            next_id = return_idx(args.path, 2)
            
            for i in range(0,32):
                filename = f"{i:02d}_rgba"
                copy_and_rename_image(args.path + '/zero123/' + filename, directory + 'ref4', f"{next_id:02d}.png")
    

    print(all_clip_score)
    last_index = []
    original_data = justify_data(args.path, args.number)

    for item in original_data:
        if item not in last_index:
            last_index.append(item)

    last_index = sorted(last_index)
    print("end")
    
    filtered_camera = filter_camera_by_indices(camera, last_index)
    camera_json = {
        "photo-nums": len(filtered_camera),
        "photos": filtered_camera
    }
    with open(args.path + "/camera.json", 'w') as json_file:
        json.dump(camera_json, json_file, indent=4)

    for i in range(0,32):
        filename = f"{i:02d}_rgba"
        dest_files = os.path.join(args.path, "zero123/", filename)
        for file in sorted(os.listdir(dest_files)):
            idx = int(file.split('.')[0])
            if idx not in last_index:
                os.remove(os.path.join(dest_files, file))

        # sort again rename
        for file in sorted(os.listdir(dest_files)):
            idx = int(file.split('.')[0])
            new_idx = last_index.index(idx)
            new_file = f"{new_idx:02d}.png"
            os.rename(os.path.join(dest_files, file), os.path.join(dest_files, new_file))
        
