import os
from PIL import Image

INPUT_DIR = os.path.join('/home/user/Documents/ai-toolkit', 'datasets', 'input')
OUTPUT_DIR = os.path.join('/home/user/Documents/ai-toolkit', 'datasets', 'output')

def process_images():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} does not exist.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Found {len(files)} files in {INPUT_DIR}")

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        has_output = os.path.exists(output_path)
        
        try:
            with Image.open(input_path) as img:
                original_w, original_h = img.size
                should_resize = original_w < 512
                
                img_to_save = img
                
                if should_resize:
                    scale = 512 / original_w
                    new_h = int(original_h * scale)
                    print(f"Resizing {filename}: {original_w}x{original_h} -> 512x{new_h}")
                    img_to_save = img.resize((512, new_h), Image.Resampling.LANCZOS)
                
                # Process specific output file if it exists
                if has_output:
                    with Image.open(output_path) as out_img:
                        out_to_save = out_img
                        if should_resize:
                            # Apply same resize
                            # We force the output to match the NEW dimensions of the input
                            # assuming they are paired.
                            out_to_save = out_img.resize((512, new_h), Image.Resampling.LANCZOS)
                        
                        save_as_jpg(out_to_save, output_path)
                
                save_as_jpg(img_to_save, input_path)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

def save_as_jpg(img, original_path):
    root, _ = os.path.splitext(original_path)
    new_path = root + '.jpg'
    
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
        
    img.save(new_path, 'JPEG', quality=95)
    
    if original_path != new_path and os.path.exists(original_path):
        os.remove(original_path)

if __name__ == '__main__':
    process_images()
