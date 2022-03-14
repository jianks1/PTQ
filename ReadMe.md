## PTQ(Post Training Quantization) Tool

这是一个对onnx模型进行PTQ量化的工具，此工具支持int8量化
/* 
This tool can be used to quantize select ONNX models. Support is based on operators in the model. 
*/


#### usage
需要安装onnxruntime

```
quantize_static(input_model_path,
                output_model_path,
                dr,
                quant_format=args.quant_format,
                op_types_to_quantize=['Conv', 'MatMul', 'Add', 'Mul', 'Relu', 'Clip'],
                per_channel=args.per_channel,
                weight_type=QuantType.QInt8,
                nodes_to_exclude=['input_1', 'input_2', 'output', 'Concat_0', 'Conv_223', 'Conv_214', 'Conv_1', 'Conv_0', 'Conv_9', 'Conv_13', 'Conv_17', 'Conv_18', 'Conv_22', 'Conv_26', 'Conv_28', 'Conv_32', 'Conv_36', 'Conv_37', 'Conv_41', 'Conv_45', 'Conv_47', 'Conv_51', 'Conv_55', 'Conv_57', 'Conv_216', 'Conv_243', 'Conv_245'],
                calibrate_method=CalibrationMethod.MinMax
                )
```


For more usage details, please refer to https://onnxruntime.ai/docs/performance/quantization.html 


#### examples
call run.py to calibrate, quantize and run the quantized model, e.g.:
python run.py --input_model mobilenetv2-7.onnx --output_model mobilenetv2-7.quant.onnx --calibrate_dataset ./test_images/

More examples please refer to https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization.


