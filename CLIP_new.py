import torch
import clip
from PIL import Image
import pyautogui
import os
import numpy as np
from pathlib import Path

# Load CLIP once
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def find_icon_by_text(text_descriptions):
    """
    Search for an icon on screen using text descriptions.
    
    Args:
        text_descriptions (list): List of text descriptions of the icon
        
    Returns:
        tuple: (x, y) coordinates of the best match
    """
    # Step 1: Analyze screen with grid
    scores, rows, cols = analyze_screen_with_grid(text_descriptions)
    
    # Step 2: Find the best match (highest score)
    # For each patch, find the maximum score across all descriptions
    max_scores_per_patch = torch.max(scores, dim=1)[0]
    best_patch_idx = torch.argmax(max_scores_per_patch).item()
    
    # Step 3: Get the coordinates of the best patch
    x, y = locate_patch_coords(best_patch_idx, rows, cols)
    
    return x, y, best_patch_idx, max_scores_per_patch[best_patch_idx].item()

def find_icon_by_template(template_path):
    """
    Search for an icon on screen using a template image.
    
    Args:
        template_path (str): Path to the template icon image
        
    Returns:
        tuple: (x, y) coordinates of the best match
    """
    # Load template image
    template = Image.open(template_path)
    template_tensor = preprocess(template).unsqueeze(0).to(device)
    
    # Take screenshot and divide into patches
    patch_size = (64, 64)  # Default patch size - should match the size used in analyze_screen_with_grid
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
    
    # Encode all images
    with torch.no_grad():
        template_features = model.encode_image(template_tensor)
        patch_features = model.encode_image(patch_batch)
        
        # Compute similarity scores
        similarities = torch.nn.functional.cosine_similarity(
            patch_features, template_features.expand(patch_features.shape[0], -1)
        )
    
    # Find best match
    best_patch_idx = torch.argmax(similarities).item()
    best_score = similarities[best_patch_idx].item()
    
    # Get coordinates
    x, y = locate_patch_coords(best_patch_idx, rows, cols)
    
    return x, y, best_patch_idx, best_score

def find_icon_in_template_folder(template_folder, text_description=None):
    """
    Find an icon on screen by checking multiple templates from a folder.
    
    Args:
        template_folder (str): Path to folder containing template images
        text_description (str, optional): Text description to filter templates
        
    Returns:
        tuple: (x, y) coordinates of the best match, template name
    """
    template_dir = Path(template_folder)
    best_match = None
    best_score = -1
    best_coords = (0, 0)
    best_template = None
    
    # Get all image files from the template folder
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    templates = [f for f in template_dir.iterdir() 
                if f.suffix.lower() in image_extensions]
    
    # Filter templates by text description if provided
    if text_description and len(templates) > 1:
        # Encode template images
        template_images = [Image.open(t) for t in templates]
        template_tensors = torch.cat([preprocess(img).unsqueeze(0) for img in template_images]).to(device)
        
        # Encode text
        text_token = clip.tokenize([text_description]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(template_tensors)
            text_features = model.encode_text(text_token)
            
            # Get scores between each template and the text
            scores = torch.nn.functional.cosine_similarity(
                image_features, text_features.expand(image_features.shape[0], -1)
            )
        
        # Sort templates by score
        sorted_indices = torch.argsort(scores, descending=True)
        templates = [templates[i] for i in sorted_indices]
    
    # Try each template
    for template_path in templates:
        x, y, _, score = find_icon_by_template(template_path)
        
        if score > best_score:
            best_score = score
            best_coords = (x, y)
            best_template = template_path.name
    
    return best_coords[0], best_coords[1], best_template, best_score

def analyze_screen_with_grid(candidate_texts):
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
        text_feats = model.encode_text(text_tokens)
        scores = (image_feats @ text_feats.T).softmax(dim=-1)

    return scores, rows, cols


def locate_patch_coords(best_patch_idx, rows, cols):
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

# Example usage:
if __name__ == "__main__":
    # 1. Find icon by text description
    # x, y, _, score = find_icon_by_text(["settings icon", "gear icon", "configuration"])
    # print(f"Found icon at coordinates: ({x}, {y}) with confidence score: {score:.4f}")
    
    # 2. Find icon by template image
    # x, y, _, score = find_icon_by_template("path/to/template/icon.png")
    # print(f"Found icon at coordinates: ({x}, {y}) with confidence score: {score:.4f}")
    
    # 3. Find icon using templates from a folder (optionally filtered by text)
    x, y, template_name, score = find_icon_in_template_folder(
        "path/to/template/folder", 
        text_description="settings icon"  # Optional
    )
    print(f"Found icon from template '{template_name}' at coordinates: ({x}, {y}) with confidence score: {score:.4f}")
    
    # You can now click on the icon using pyautogui
    # pyautogui.click(x, 