import torch
import clip
from PIL import Image
import pyautogui
from langchain.tools import tool
# Load CLIP once
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

@tool
def analyze_screen_with_grid(candidate_texts: str):
    """
    Divide the screen into patches, run CLIP on each patch vs each candidate,
    and return a scores tensor of shape (num_patches, num_candidates), plus
    the grid dimensions rows, cols.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    patch_size = (64, 64)  # Default patch size
    screen_image = pyautogui.screenshot()
    w, h = screen_image.size
    cols = max(1, w // patch_size[0])
    rows = max(1, h // patch_size[1])

    patches = []
    for r in range(rows):
        for c in range(cols):
            left, top = c*patch_size[0], r*patch_size[1]
            patch = screen_image.crop((left, top,
                                       left+patch_size[0],
                                       top+patch_size[1]))
            patches.append(preprocess(patch).unsqueeze(0))

    patch_batch = torch.cat(patches).to(device)
    text_tokens = clip.tokenize(candidate_texts).to(device)

    with torch.no_grad():
        image_feats = model.encode_image(patch_batch)
        text_feats  = model.encode_text(text_tokens)
        scores = (image_feats @ text_feats.T).softmax(dim=-1)

    return scores, rows, cols


@tool
def locate_patch_coords(best_patch_idx: int,
                        rows: int, cols: int) -> tuple[int,int]:
    """
    Map a patch index back to the center (x,y) pixel in the full screenshot.
    """
    screen_img = pyautogui.screenshot()
    w, h = screen_img.size
    patch_w = w / cols
    patch_h = h / rows
    r, c = divmod(best_patch_idx, cols)
    x = int((c + 0.5) * patch_w)
    y = int((r + 0.5) * patch_h)
    return x, y
