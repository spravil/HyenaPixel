import argparse

import hyenapixel.models
import numpy as np
import timm
import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape


def op_handle_fft(inputs, outputs):
    output_shape = get_shape(outputs[0])
    return {"fft": np.prod(output_shape) * np.log2(np.prod(output_shape[2:]))}


def main(model_name):
    model = timm.create_model(model_name, pretrained=False)
    input = torch.zeros(1, 3, 224, 224)
    macs = (
        FlopCountAnalysis(model, input)
        .set_op_handle("aten::fft_rfft2", op_handle_fft)
        .set_op_handle("aten::fft_irfft2", op_handle_fft)
        .set_op_handle("aten::fft_rfft", op_handle_fft)
        .set_op_handle("aten::fft_irfft", op_handle_fft)
    )
    print(macs.total() / 1_000_000_000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MACs for a given model")
    parser.add_argument(
        "--model_name", type=str, default="hpx_former_b18", help="Name of the model to calculate MACs for"
    )
    args = parser.parse_args()
    main(args.model_name)
