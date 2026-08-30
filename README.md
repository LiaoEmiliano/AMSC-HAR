# AMSC-HAR: Video Based Human Action Recognition

AMSC-HAR is a header-only C++23 library and command-line trainer for video-based human action recognition. It was developed as part of the Advanced Methods for Scientific Computing (AMSC) course at Politecnico di Milano. 

Unlike typical projects that rely on PyTorch or TensorFlow, this system implements tensors, neural network layers, automatic reverse-mode gradients, SGD optimization, checkpointing, video decoding, and optional CUDA kernels **from scratch**. 

The core model is a compact **VideoCNN**: a shared 2D convolutional backbone over sampled RGB frames, followed by a residual temporal 1D convolution, learnable temporal attention pooling, and a linear classifier. 

## Features
- **Generic Tensor Library**: A custom `Tensor<T>` class with CPU storage and optional CUDA Unified Memory.
- **Layers with Explicit Gradients**: `Conv2D`, `BatchNorm2D`, `ReLU`, Pooling, `Dropout`, `Linear`, `TemporalConv1D`, and Temporal Attention.
- **Data Pipeline**: End-to-end decoding using FFmpeg/OpenCV, clip caching, and temporal sampling for efficient training.
- **Optimized for Speed**: Dual CPU/CUDA paths allowing the same headers to train on laptops or GPUs.

## Dependencies
- **C++23** compiler (Apple Clang, GCC, or MSVC)
- **CMake** >= 3.25
- **FFmpeg** (must be available in PATH for video decoding)
- *(Optional)* **CUDA Toolkit** >= 13.x for GPU acceleration (requires an NVIDIA GPU).
- *(Optional)* **OpenCV** (if available, used for faster video decoding instead of FFmpeg).

## Build Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LiaoEmiliano/AMSC-HAR
   cd AMSC-HAR
   ```

2. **Build the project using CMake:**
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```
   The compiled executable will be located at `build/har_cnn`.

## Downloading Datasets

The project natively supports the **UCF11 (YouTube Action)** and **UCF101** datasets.
You can download and extract UCF11 automatically using the provided PowerShell script (Windows) or curl/unrar (Unix-like):

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_ucf11.ps1
```

**macOS/Linux:**
```bash
brew install rar # or equivalent package manager
mkdir -p data
curl -L "https://www.crcv.ucf.edu/data/UCF11_updated_mpg.rar" -o data/UCF11_updated_mpg.rar
unrar x -y data/UCF11_updated_mpg.rar data/
```

## Usage

You can use the compiled `har_cnn` executable to train, evaluate, or run predictions.

### 1. Predict / Inference
To classify new videos, place `.mpg` or `.mp4` files into a directory (e.g., `eval_videos`) and run:
```bash
./build/har_cnn predict --dir eval_videos --weights Weights/ucf11_videocnn.harw
```
*(By default, it uses the provided checkpoint in `Weights/ucf11_videocnn.harw` and prints the Top-3 predictions).*

### 2. Training
To train the VideoCNN on the UCF11 dataset:
```bash
./build/har_cnn train --dataset ucf11 --data data/UCF11_updated_mpg --epochs 10 --batch 32
```
Use `--help` to see all hyperparameters (learning rate, momentum, dropout, size, frames, etc.).

### 3. Evaluation
Evaluate a trained model checkpoint against a dataset split (e.g. testing set):
```bash
./build/har_cnn eval --dataset ucf11 --data data/UCF11_updated_mpg --weights Weights/ucf11_videocnn.harw
```

### 4. Extract Frames
Extract static frames from a video file manually using the backend decoder:
```bash
./build/har_cnn extract <video_path> [output_directory]
```

## Project Structure

- `include/har/` : Header-only C++23 library (Tensors, Layers, Optimizers, Network, Data).
- `src/` : CUDA kernels and CPU source entry points.
- `main.cpp` : The CLI application parsing arguments and routing commands.
- `Weights/` : Checkpoint (`.harw`) files for pretrained models.
- `report/` : LaTeX source and PDF containing the mathematical modeling, architecture details, and experimental results.
- `third_party/` : Vendored dependencies (stb_image, etc.).