"""
    BP-nnUNet based on: "Kwon et al.,
    Blood Pressure Assisted Cerebral Microbleed Segmentation via Meta-matching
    <https://link.springer.com/chapter/10.1007/978-3-032-04927-8_8>"

    Part of the codes are referred from:
    nnU-Net based on: "Isensee et al.,
    nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
    <https://www.nature.com/articles/s41592-020-01008-z>"
"""
from pathlib import Path
import torch
BP_DICT = {}
EMBED_DICT = {}

def _load_embedding_pair(embed_path, dataset_name):
    mapper = {
        "bcp": f"{dataset_name}-BCP",
        "nobcp": f"{dataset_name}-noBCP",
    }
    embeddings = {}
    for key_name, filename in mapper.items():
        path = Path(embed_path) / f"{filename}.pth"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing text embedding: {path}. Run "
                f"Code/biomed_clip/init_biomedclip_textembed.py first."
            )
        embed = torch.load(path, map_location="cpu").reshape(1, 512)
        if torch.cuda.is_available():
            embed = embed.cuda()
        embeddings[key_name] = embed
    return embeddings

# Read the text embeddings from the embedding directory
def get_embeddings(embed_path=None, dataset_name="LA"):
    if dataset_name not in EMBED_DICT:
        if embed_path is not None:
            search_paths = [Path(embed_path)]
        else:
            search_paths = [
                Path(__file__).resolve().parents[2] / "biomed_clip" / "embeddings",
                Path(
                    "/home/mengqingxu/Project_MQX/MyProj/9_SemiGeneralEXP/"
                    "103_BiomedCLIP/Code/biomed_clip/embeddings"
                ),
            ]

        last_error = None
        for candidate in search_paths:
            try:
                EMBED_DICT[dataset_name] = _load_embedding_pair(candidate, dataset_name)
                break
            except FileNotFoundError as exc:
                last_error = exc
        else:
            raise last_error
    return EMBED_DICT[dataset_name]