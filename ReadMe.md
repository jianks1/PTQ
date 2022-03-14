## PTQ(Post Training Quantization) Tool

这是一个对onnx模型PTQ量化工具quantize_static的中文版说明，此工具支持int8量化


#### usage
需要安装onnxruntime，量化工具处于文件夹onnxruntime/python/tools/quantization/下面，这里主要说明工具中的静态量化API quantize_static，给定一个onnx模型和校准数据阅读器，quantize_static创建一个量化的onnx模型并将其保存到一个文件中，它的调用参数如下：

```
quantize_static(model_input,
                model_output,
                calibration_data_reader: CalibrationDataReader,
                quant_format=QuantFormat.QOperator,
                op_types_to_quantize=[],
                per_channel=False,
                reduce_range=False,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
                nodes_to_quantize=[],
                nodes_to_exclude=[],
                optimize_model=True,
                use_external_data_format=False,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options = {}
                )
```

:param model_input:量化模型的文件路径
:param model_output:量化模型的文件路径
:param calibration_data_reader:校准数据阅读器。它枚举校准数据并生成原始模型的输入。
:param quant_format: QuantFormat{QOperator, QDQ}。QOperator格式直接用量化算子量化模型。QDQ格式通过在张量上插入quantizellinear / dequantizellinear来量化模型。
:param op_types_to_quantize:指定要量化的操作符的类型，如op_types_to_quantize=['Conv', 'MatMul', 'Add', 'Mul', 'Relu', 'Clip']，只量化'Conv', 'MatMul', 'Add', 'Mul', 'Relu', 'Clip'。默认情况下，它量化所有支持的操作符。
:param op_types:量化操作符
:param per_channel:量化每个通道的权重
:param reduce_range:量化7位的权重。它可以提高一些运行在非vnni机器上的模型的准确性，特别是对于每通道模式
:param activation_type:激活的量化数据类型
:param weight_type:权重的量化数据类型
:param nodes_to_quantize:要量化的节点名称列表。当该列表不是None时，只有该列表中的节点被量化。示例:nodes_to_exclude=['input_1', 'input_2', 'output', 'Concat_0', 'Conv_223', 'Conv_1', 'Conv_0', 'Conv_9', 'Conv_13', 'Conv_17', 'Conv_18', 'Conv_22', 'Conv_26', 'Conv_28', 'Conv_32', 'Conv_36', 'Conv_37', 'Conv_41', 'Conv_45', 'Conv_47', 'Conv_51', 'Conv_55', 'Conv_57', 'Conv_216', 'Conv_243', 'Conv_245'],
:param nodes_to_exclude:要排除的节点名称列表。当列表不为None时，该列表中的节点将被排除在量化之外。
:param optimize_model:量化之前优化模型。:参数use_external_data_format:用于大尺寸(&gt;2GB)模型的选项。默认为False。
:param calibrate_method:当前支持的校准方法有MinMax和Entropy。请使用CalibrationMethod。极大极小或CalibrationMethod。熵作为选项。
:param extra_options:不同情况下不同选项的键值对字典。

针对于模型的权重，该工具采用不饱和对称量化，其计算scale和zero_point的核心代码如下：
```
def compute_scale_zp(rmin, rmax, qmin, qmax, symmetric=False):

    rmin = min(rmin, 0)
    rmax = max(rmax, 0)

    if symmetric:
        absmax = max(abs(rmin), abs(rmax))
        rmin = -absmax
        rmax = +absmax

    scale = (rmax - rmin) / float(qmax-qmin) if rmax!=rmin else 1.0
    zero_point = round(qmin - rmin/scale)

    return [zero_point, scale]
```
该函数首先统计权值数据中的最大最小值，然后得到其得到其最大的绝对值，该值的正负值作为边缘值对应到要量化到的最大最小值，采用int8量化时为127和-127，然后即可计算得到scale，zero_point为0。

For more usage details, please refer to https://onnxruntime.ai/docs/performance/quantization.html 


#### examples

需要准备校准数据，该数据用于收集模型forward过程中的激活值，以便对其进行量化

python run.py --input_model mobilenetv2-7.onnx --output_model mobilenetv2-7.quant.onnx --calibrate_dataset ./test_images/

More examples please refer to https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization.


