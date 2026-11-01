from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input=r"C:\Users\Mikey\runs\detect\train\weights\best_sim.onnx",
    model_output=r"C:\Users\Mikey\runs\detect\train\weights\dynamic_int8.onnx",
    weight_type=QuantType.QUInt8
)