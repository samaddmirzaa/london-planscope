'''
Export the fine-tuned classifier to ONNX for fast CPU serving, plus an int8-quantised copy.
Writes models/category-onnx/: model.onnx, model.int8.onnx, tokenizer.json, labels.json.

'''

import json
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'models' / 'category-distilbert'
ONNX_DIR = ROOT / 'models' / 'category-onnx'


def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

    # Trace the model on an example batch; batch size and text length may vary at prediction time
    example = tokenizer(['single storey rear extension', 'prune 1 x oak tree'], padding=True, return_tensors='pt')
    batch, tokens = torch.export.Dim('batch'), torch.export.Dim('tokens', max=512)
    torch.onnx.export(
        model,
        (example['input_ids'], example['attention_mask']),
        ONNX_DIR / 'model.onnx',
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_shapes={'input_ids': {0: batch, 1: tokens}, 'attention_mask': {0: batch, 1: tokens}},
        external_data=False,
    )

    # Drop the exporter's shape annotations (they confuse the quantiser), then quantise weights to int8
    graph = onnx.load(ONNX_DIR / 'model.onnx')
    graph.graph.ClearField('value_info')
    onnx.save(graph, ONNX_DIR / 'model.onnx')
    quantize_dynamic(ONNX_DIR / 'model.onnx', ONNX_DIR / 'model.int8.onnx', weight_type=QuantType.QInt8)

    # The tokenizer and category names, so serving needs neither PyTorch nor transformers
    tokenizer.backend_tokenizer.save(str(ONNX_DIR / 'tokenizer.json'))
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    (ONNX_DIR / 'labels.json').write_text(json.dumps(labels))
    for path in sorted(ONNX_DIR.iterdir()):
        print(f'{path.name:18} {path.stat().st_size / 1e6:6.1f} MB')


if __name__ == '__main__':
    main()