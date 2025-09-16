# TAPOR: 3D Hand Pose Reconstruction with Fully Passive Thermal Sensing for Around-Device Interactions

TAPOR is a privacy-preserving, non-contact sensing system that enables accurate and robust 3D hand pose reconstruction for around-device interaction using a single low-cost thermal array sensor. By leveraging inexpensive and miniature thermal arrays, TAPOR achieves an excellent balance between utility and privacy, offering imaging capabilities that bridge the gap between high-resolution cameras and low-resolution RF signals.

![Demo](figures/short_demo1(1).gif)

[📺 Watch Full Demo with Applications](https://www.youtube.com/watch?v=dRiqxPZx4zk)

## Key Features
- Privacy-preserving hand pose reconstruction using passive thermal sensing
- Novel physics-inspired neural network for effective 3D spatial representation
- Efficient deployment on IoT devices through heterogeneous knowledge distillation (377× computation reduction)
- Robust performance in real-world scenarios
- Support for gesture control and finger tracking applications

## Getting Started

### Environment Setup
```bash
# For TAPOR main environment
conda create -f environment.yml

# For NanoTapor development (optional)
conda create -f environment2.yml
```

### Dataset and Pre-trained Models
Download the following resources and place them in the root directory:
- [Training Dataset](https://drive.google.com/file/d/12Ppoh15nomnVAKo2UFQRqgcLeDRh5hGk/view?usp=sharing)
- [Test Dataset](https://drive.google.com/file/d/1cnwdelIiV4V15rq3ANN93YsNvAjpvV5O/view?usp=sharing)
- [Pre-trained Weights](https://drive.google.com/file/d/183gpLbMAORcaPqkF7naI5xlrQ5T2p_MJ/view?usp=sharing)

*Please download the 'Mano models' from https://drive.google.com/file/d/1X1DpN1p6bHkmlZn9fDXc-vT12hSp9hWT/view?usp=sharing and add the files into models/mano_base/mamo/*

## Model Training and Evaluation

### Training TAPOR
```python
# Train the main TAPOR model
python traintest.py -m tapor -e 200 -b 48 -fs 1 -ms 0 -t 0 -lr 0.0001 -hm 0 -fo 1 -s 0 -mt 1 -ls jb -tqdm 1 

# Train baseline models for comparison
python traintest.py -m mediapipe -e 200 -b 64 -fs 1 -ms 256 -t 0 -lr 0.001 -hm 0 -fo 0 -s 2 
python traintest.py -m baseline3d -e 400 -b 64 -fs 1 -ms 200 -t 0 -lr 0.00001 -hm 1 -fo 1 -s 3 
python traintest.py -m mano -e 200 -b 64 -fs 1 -ms 200 -t 0 -lr 0.0001 -hm 0 -fo 1 -s 0 
```

Training outputs:
- Tensorboard logs → 'Runs' folder
- Model weights → 'weights' folder
- Prediction logs → 'LogTrainTest' folder (pickle format)

### Model Evaluation
```python
# Evaluate TAPOR
python inference.py -m tapor -wp tapor.pth -fs 1 -ms 0 -hm 0 -fo 1 -mt 1 -v 0 

# Evaluate baseline models
python inference.py -m mano -wp mano.pth -fs 1 -ms 200 -hm 0 -fo 1 -v 0 
python inference.py -m mediapipe -wp mediapipe.pth -fs 1 -ms 256 -hm 0 -fo 1 -v 0 
python inference.py -m baseline3d -wp baseline3d.pth -fs 1 -ms 200 -hm 0 -fo 1 -v 0 
```

Evaluation results will be saved in the 'TestDataset_logs' folder.

## IoT Deployment with NanoTapor
Please use the Firmware folder to flash the NanoTapor firmware on the ESP32S3 device. 

### NanoTapor Development Pipeline (optional)

1. **Prepare Training Data**
   - Complete TAPOR model training or download pre-trained weights
   - Run `NanoTapor_trainset_construction.ipynb` to generate training data
   - Output: `KD_training_data.pkl` in 'NanoTapor_files/'

2. **Train NanoTapor**
   ```python
   python NanoTapor_train.py -wp "tapor.pth" -fs 1 -mt 1 -fd 336 -ld 64 -ft 0 -lr 0.00001 -bs 32 -ep 100 -dd 0 -v 0 
   ```

3. **Evaluate NanoTapor**
   ```python
   python NanoTapor_test.py -wp "24_202436211552_tm2.pth" -fs 1 -fd 336 -ld 48 -bs 128 -dd 0 -v 0
   ```

4. **Deploy to ESP32**
   - Convert model: Run `NanoTapor_pytorch2onnx.ipynb` and `NanoTapor_onnx2tf_lite.ipynb`
   - Copy generated hex model from `testing_model_data.cc` to `Firmware/src/model_data.cc`
   - Flash firmware to ESP32S# device
   
   Pre-generated deployment files available [here](https://drive.google.com/file/d/1qX8qTc5dIomArVZT7GhbGCIVZ4NH6nFY/view?usp=sharing)

## Gesture Recognition

### Dataset
Download the gesture recognition dataset [here](https://drive.google.com/file/d/1JmTD0N_9hdhCS8JOSd9t2bQ_NmB9BbdB/view?usp=sharing)

### Training and Evaluation
```python
# Train and test TAPOR for gesture recognition
python GR_traintest.py -wp "tapor.pth" -fs 1 -ms 0 -mt 1 -ev 1 -ptm 0 -lr 0.0001 -bs 32 -ep 100 -dd 0

# Train and test NanoTapor for gesture recognition
python GR_traintest.py -wp "44_20243715330_tm2_adaptor.pth" -fs 1 -ms 0 -mt 8 -ev 4 -ptm 1 -lr 0.0001 -bs 32 -ep 1000 -dd 0
```

Results will be saved in the 'gesture_logs' folder.

## Paper Results Reproduction
- Check `z_results_visualization.ipynb` for figure generation
- Download experimental results [here](https://drive.google.com/file/d/1Q_BTcYwpz4F0oT0bDqio4qohWCSYecxe/view?usp=sharing)

## License
MIT License

## Citation
```
@article{Zhangtapor2025,
  title = {{{TAPOR}}: {{3D}} Hand Pose Reconstruction with Fully Passive Thermal Sensing for around-Device Interactions},
  author = {Zhang, Xie and Li, Chengxiao and Wu, Chenshu},
  year = {2025},
  month = jun,
  journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  volume = {9},
  number = {2},
  doi = {10.1145/3729499},
  articleno = {63}
}
```
