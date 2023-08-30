import os.path as osp
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

test_arr = np.random.randn(1, 3, 480, 480).astype(np.float32)

# model = torch.load(r'F:\Artificial_neural_Network\yolov8-main\weights\yolov8n.pt').cuda().eval()
# print('pytorch result:', model(torch.from_numpy(test_arr).cuda()))

model_onnx = onnx.load(r'F:\Artificial_neural_Network\anomalib-main\results\patchcore\membrane\run\weights\onnx\model.onnx')
onnx.checker.check_model(model_onnx)

ort_session = ort.InferenceSession(r'F:\Artificial_neural_Network\anomalib-main\results\patchcore\membrane\run\weights\onnx\model.onnx')
outputs = ort_session.run(None, {'input': test_arr})
print('onnx_result:', outputs)
print(outputs)