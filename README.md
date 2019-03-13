# Mobilenet V1 for STM32 over CMSIS-NN

This project contains a STM32 Workbench project to run a Mobilenet v1 ('160x160x3', alpha '0.25') on a STM32H7 NUCLEO evaluation board.

## What is a Mobilenet?
[MobileNets](https://arxiv.org/abs/1704.04861) are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models, such as Inception, are used. MobileNets can be run efficiently on mobile devices with [TensorFlow Mobile](https://www.tensorflow.org/mobile/).
MobileNets trade off between latency, size and accuracy while comparing favorably with popular models from the literature.

![alt text](https://github.com/tensorflow/models/raw/master/research/slim/nets/mobilenet_v1.png "MobileNet Graph (Credits https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)")

The parameters used to meet the memory constraints imposed by the STM32H7 are: image input size is '160x160x3', alpha '0.25'.
The following table show the classification performance of such configuration.

Model  | Million MACs | Million Parameters | Top-1 Accuracy| Top-5 Accuracy |
:----:|:------------:|:----------:|:-------:|:-------:|
[MobileNet_v1_0.25_160_uint8](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_160_quant.tgz)|21|0.47|43.4|68.5|

## Asymmetric UINT8 CMSIS-NN support
To limit the classification error introduced by the quantization this mobilenet implementation uses an extended [CMSIS-NN](https://github.com/EEESlab/CMSIS_NN-INTQ) that support an aymmetric quantization methodology mapped on UINT8 datatypes.
You find additional information on this [article](https://medium.com/@manuele.rusci/running-mobilenet-on-stm32-mcus-at-the-edge-e217db934f83).

## Howto import a Tensorflow model
Shortly will be release a Python script for the import of pre-trained Tensorflow models.

## Getting-Started on STM Workbench
+ Get a STMicroelectronics [NUCLEO STM32H743ZI](https://www.st.com/en/evaluation-tools/nucleo-h743zi.html)
+ Install [System Workbench for STM32](https://www.st.com/en/development-tools/sw4stm32.html) for your OS and open it
+ Click 'File> Import...', then select 'Git> Project from Git' and click 'Next'

![alt text](docs/import1.jpg)
+ Select 'Clone URI' and click 'Next'

![alt text](docs/import2.jpg)
+ Add the URI 'https://github.com/EEESlab/mobilenet_v1_stm32_cmsis_nn.git' and click 'Next'

![alt text](docs/import3.jpg)
+ Select all the branches and click 'Next'

![alt text](docs/import4.jpg)
+ Select the local destination and click 'Next'

![alt text](docs/import5.jpg)
+ Select how to import the project and click 'Next'

![alt text](docs/import6.jpg)
+ Import the project (finally)!

![alt text](docs/import7.jpg)

+ Build and run the application!

## Measured Performance
The following table show the performance and the profiling layer-per-layer of this mobilenet implementation.
![alt text](docs/profiling_160_0_25.png)



