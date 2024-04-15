## Implementation Log

### 18/07/30
- Commit initial implementation of PoseNet
  - Base architecture: Google (inception v3) + regressor (position and orientation)
    - No mid fully connected layer (2048)
  - Works fine with following paramerters
    - Batch size: 4 / lr: 1e-4
  - Test accuracy getting worse when use large batch size

### 18/08/01
- Add Resnet-34 for base architecture
- Survey various control paramerters (Each case show good results on training and validation process, but not for the test)
  - Batch size: large batchsize -> weak generalization?
  - add 2048 fully connected layer -> Training not converge well
  - Initialization: Kaming Normal -> Normal
  - Weight decay: 0.0625? 0.0005?
  - Learning rate & decay: small batchsize (large lr available)
  - Shuffle data
- Curretly, the best performance
  - Pretrained Resnet 34 - fully(512 to 7, no bias)
    - batch size: 4

### 18/08/07
- Add Bayesian Posenet mode, but not tested yet
- Still, cannot find the reason on weak performance compared to other implementations
- Weight initialization seems important for training
  - Add weight initialization followed by the original caffe-posenet code

### 18/08/08
- Overall training loss do not decrease as we expected
  - Rotation loss is not trained well
- Small batchsize(=4) seems work fine
  - Code bug in test mode (did not model.eval())

### 18/08/09
- Now fixed bug in the test code.
  - Previous training process worked fine, there was an bug in the test Code
    - model.eval() affects the performance severly
  - Tested model
    - batch size: 4, 8, 16, 64 (All converge well)
    - initializatio: kaming normal, normal, none (All converge well)
    - validation: crop training data (worked fine but little low performance)
    - Loss function: L1, MSE, L2 (All worked fine)
