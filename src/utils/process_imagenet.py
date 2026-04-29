import os
import random
import shutil
from pathlib import Path

random.seed(42)

# Dictionary with the classes and the exact quantity required by the article
target_classes = {
    'train': {'goldfish': 648, 'pineapple': 520, 'clock': 455},
    'validation': {'jellyfish': 618},
    'test': {'zebra': 503, 'panda': 501, 'cab': 485}
}

origin_path = Path('./data/imagenet/raw')
destination_path = Path('./data/imagenet/subset')

def process_dataset():
    for split, classes in target_classes.items():
        for name_class, quantity in classes.items():
            origin_folder = origin_path / name_class
            destination_folder = destination_path / split / name_class
            
            destination_folder.mkdir(parents=True, exist_ok=True)
            
            # Get all valid images (ignoring hidden/corrupted files)
            imagens = [f for f in os.listdir(origin_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(imagens) < quantity:
                print(f"ERRO: The class {name_class} has only {len(imagens)} images. We need {quantity}.")
                continue
                
            # Selection with aleatory seed for reproducibility
            imagens_selecionadas = random.sample(imagens, quantity)
            
            for img in imagens_selecionadas:
                shutil.copy2(origin_folder / img, destination_folder / img)
                
            print(f"[{split.upper()}] {quantity} images of {name_class} were copied.")

if __name__ == '__main__':
    process_dataset()
    print("Selection completed successfully!")