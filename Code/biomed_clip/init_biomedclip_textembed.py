"""
    BP-nnUNet based on: "Kwon et al.,
    Blood Pressure Assisted Cerebral Microbleed Segmentation via Meta-matching
    <https://link.springer.com/chapter/10.1007/978-3-032-04927-8_8>"

    Part of the codes are referred from:
    CLIP-Driven Universal Model based on: "Liu et al.,
    CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection
    <https://ieeexplore.ieee.org/document/10376801>"
"""

from pathlib import Path
import open_clip
import torch

from prompt_ablation import PROMPT_ABLATIONS

# 本地 BiomedCLIP 目录
# 该目录下建议包含：
# open_clip_config.json
# open_clip_pytorch_model.bin
# tokenizer_config.json / vocab.txt / config.json 等 tokenizer 文件
MODEL_DIR = Path("./pth")

OUTPUT_DIR = Path("./embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 方式 1：推荐，新版 open_clip 支持 local-dir
model, _ = open_clip.create_model_from_pretrained(
    f"local-dir:{MODEL_DIR}",
)

tokenizer = open_clip.get_tokenizer(
    f"local-dir:{MODEL_DIR}"
)


model = model.to(device)
model.eval()


INFO = {}

for variant, config in PROMPT_ABLATIONS.items():
    INFO[f"{variant}-BCP"] = [config["bcp"]]
    INFO[f"{variant}-noBCP"] = [config["nobcp"]]

for filename, description in INFO.items():
    with torch.no_grad():
        text_inputs = tokenizer(description).to(device)
        text_embed = model.encode_text(text_inputs, normalize=True)
        # 不建议硬编码 reshape(1, 512)，避免后续换模型维度时报错
        text_embed = text_embed.reshape(text_embed.shape[0], -1)
        torch.save(text_embed.cpu(), OUTPUT_DIR / f"{filename}.pth")
