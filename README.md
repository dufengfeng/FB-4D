<h1>FB-4D: Training-free Dynamic 3D Content Generation with Feature Banks</h1>

# Installation
```bash
pip install -r requirements.txt
git clone --recursive https://github.com/slothfulxtx/diff-gaussian-rasterization.git
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
```
# Dataset Preparing

***Test dataset*** can be downloaded from [google drive](https://cloud.tsinghua.edu.cn/d/9b6bad311f7d42a387f8/).(Our final results are also included)
The file should be organized as follows:
```
├── dataset
│   | object-1
│     ├── 00_rgba.png
│     ├── 01_rgba.png
│     ├── 02_rgba.png
│     ├── ...
│   | object-2
│     ├── 00_rgba.png
│     ├── 01_rgba.png
│     ├── 02_rgba.png
│     ├── ...
│   | ...
```
If you'd like to generate more 4D objects, you can organize your data in the same format as above. 

Additionally, we utilized [STAG4D](https://github.com/zeng-yifei/STAG4D) dataset to process more objects. You can download the dataset and reorganize the data as above to generate more 4D objects.

# Pipeline

**STAGE-1:** multi-view image sequences generation

you can run the following command to generate multi-view image sequences for your data:

```
CUDA_VISIBLE_DEVICES=0 python scripts/FB_gen.py --path your_path --pipeline_path your_pipeline_path --number 18
```

The parameter `number` specifies the number of iterations and can be adjusted based on your requirements (e.g., `6`, `12`, `18`). 

- **6**: Represents one iteration, generating up to six different views of image-sequences.  
- **12**: Corresponds to two iterations, producing up to twelve different views.  
- **18**: Indicates three iterations, resulting in up to eighteen different views.  


**STAGE-2:** 4D object generation

After generating multi-view image sequences, you will see a file named "camera.json" in your path and you can run the following command to generate 4D objects:

```
CUDA_VISIBLE_DEVICES=0 python main_4D.py --config configs/FB-4D.yaml path=your_path save_path=your_save_path --camera your_path/camera.json
```

# Evaluation
To avoid environment conflicts, we recommend running the evaluation code in a new conda environment. We provide all the dependicies in the `requirements_eval.txt` file. You can create a new environment and install the dependencies using the following command:
```
pip install -r requirements_eval.txt
```
We use the [Consistent4D](https://github.com/yanqinJiang/Consistent4D) to do evaluation. To evaluate, first transform rgba gt images to images with white background. Download the pre-trained model (i3d_pretrained_400.pt) for calculating FVD [here](https://drive.google.com/file/d/1J8w3fGj6H6kmcW9G8Ff6tRQofblaG5Vn/view) (This link is borrowed from DisCo, and the file for calculating FVD is a refractoring of their evaluation code. Thanks for their work!). Organize the reuslt folder as follows:
```
├── eval_dataset
      | gt
      │   ├── object_0
      │   │   ├── eval_0
      │   │   │   ├── 0.png
      │   │   │   └── ...
      │   │   ├── eval_1
      │   │   │   └── ...
      │   │   └── ...
      │   ├── object_1
      │   │   └── ...
      │   └── ...
      | pred
      │   ├── object_0
      │   │   ├── eval_0
      │   │   │   ├── 0.png
      │   │   │   └── ...
      │   │   ├── eval_1
      │   │   │   └── ...
      │   │   └── ...
      │   ├── object_1
      │   │   └── ...
      │   └── ...
```
Next, run the following command to evaluate the generated 4D objects:
```
cd metrics
# image-level metrics
CUDA_VISIBLE_DEVICES=0 python compute_image_level_metrics.py --gt_root path/gt --pred_root path/pred --output_file save_path
# video-level metrics
CUDA_VISIBLE_DEVICES=0 python compute_fvd.py --gt_root path/gt --pred_root path/pred --model_path path/i3d_pretrained_400.pt --output_file save_path
```
# Acknowledgment
This repo is built on [STAG4D](https://github.com/zeng-yifei/STAG4D). Thank all the authors for their great work.