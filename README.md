# GoSPA: A Sparse CNN Accelerator in SystemVerilog

A from-scratch SystemVerilog implementation of **GoSPA** (Deng *et al.*,*GoSPA: An Energy-efficient High-performance Globally Optimized SParse
Convolutional Neural Network Accelerator*, ISCA 2021). This repo includes a PyTorch
functional model, A Python performance model, A SystemVerilog implementation, cocotb/Verilator
verification test benches, and an FPGA synthesis flow running
real quantized MobileNetV2 and AlexNet layers end to end.

![GoSPA architecture](docs/GoSPA.jpg)

