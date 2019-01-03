# RL-fooling-framework

## preparation

download Mnist dataset and unzip all so the folder looks like:

- dataset
    - original
        - t10k-images.idx3-ubyte
        - t10k-labels.idx1-ubyte
        - train-images.idx3-ubyte
        - train-labels.idx1-ubyte

- pretrained model
    - model
        - mnist
            - checkpoint
                - LeNet
        - cifar10
            - checkpoint
                - MobileNet
                - VGG19
                
## run experiment

- To test the fooling, try ```python src/run.py 0```
- To convert a trajectory, try ```python src/trajectory_convert.py $Trajectory_location```