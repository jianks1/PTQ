## PTQ(Post Training Quantization) Tool

这是一个对onnx模型进行PTQ量化的工具，此工具支持int8量化
// This tool can be used to quantize select ONNX models. Support is based on operators in the model. 


#### usage
需要安装onnxruntime
For more usage details, please refer to https://onnxruntime.ai/docs/performance/quantization.html 


#### examples
call run.py to calibrate, quantize and run the quantized model, e.g.:
python run.py --input_model mobilenetv2-7.onnx --output_model mobilenetv2-7.quant.onnx --calibrate_dataset ./test_images/

More examples please refer to https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization.


