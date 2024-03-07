# -*- coding:utf-8 -*-

import argparse
import os
import traceback

from funasr import AutoModel

path_asr = "tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
path_vad = "tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
path_punc = "tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
path_asr = (
    path_asr
    if os.path.exists(path_asr)
    else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)
path_vad = (
    path_vad
    if os.path.exists(path_vad)
    else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
)
path_punc = (
    path_punc
    if os.path.exists(path_punc)
    else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
)

model = AutoModel(
    model=path_asr,
    model_revision="v2.0.4",
    vad_model=path_vad,
    vad_model_revision="v2.0.4",
    punc_model=path_punc,
    punc_model_revision="v2.0.4",
)


def asr_to_file(input_file, output_file):
    try:
        text = model.generate(input=input_file)[0]["text"]
        print(f"asr_to_file {input_file} {output_file} {text}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
    except:
        text = ""
        print(traceback.format_exc())
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the WAV file.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Output folder to store transcriptions.",
    )

    cmd = parser.parse_args()
    asr_to_file(input_file=cmd.input_file, output_file=cmd.output_file)
