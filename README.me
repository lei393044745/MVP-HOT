# MVP-HOT
---
## Introduction

We attempted to use the prompt learning method on three different spectral datasets.
We will soon release the model file.

## Prerequisites

Before you can run the code in this repository, you'll need to have the following installed on your system:

- Python 3.7.16
- PyTorch 1.13.1
- CUDA 11.4

## Dataset
The Hyperspectral Object Tracking Challenge 2023 [HOT023](www.hsitracking.com) offers comprehensive datasets comprising 109 training videos and 87 validation videos. These datasets were captured using three XIMEA snapshot cameras, including VIS, NIR, and RedNIR, and covering 16 bands, 25 bands, and 15 bands respectively.These videos were captured at 25 FPS. Each video consist of two types of data: hyperspectral video data and false-color video data synthesized from hyperspectral video sequences.

## Training

```
change ./SeqTrack/lib/train/admin/local.py line 26 dataset path
change ./SeqTrack/lib/train/base_functions.py line 78 --options "HSI-VIS", "HSI-NIR", "HSI-RedNIR"
change ./SeqTrack/lib/models/seqtrack/vit.py line 298 --options 16 25 15
change ./SeqTrack/lib/train/train_script.py line 61 pre_train weight path
python -m torch.distributed.launch --nproc_per_node 4 lib/train/run_training.py --script seqtrack --config seqtrack_b256 --save_dir .
```

## Testing

```
change ./SeqTrack/lib/test/tracker/prompt_seqtrack.py line 188 data_type --options "vis-" "nir-" "rednir-"
change ./SeqTrack/lib/test/tracker/eval.py line 385 386 change the video types
cd ./SeqTrack/lib/test/prompt_seqtrack.py
```

## Acknowledgments
The authors would like to express their sincere gratitude to [HOT2023](www.hsitracking.com) organizers for kindly sharing the dataset, which has been instrumental in the conduct of this research.
The code in this repository is based on [SeqTrack](https://github.com/microsoft/VideoX/tree/master/SeqTrack). Thank you to them for sharing the code.
