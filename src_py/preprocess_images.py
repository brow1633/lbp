import os
from PIL import Image

def resize_and_save_images(input_folder, output_folder, size=(256, 256)):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg')):
            # Split the filename to get the noun and the number
            noun_part = ''.join([c for c in filename if not c.isdigit()]).split('.')[0]
            number_part = ''.join([c for c in filename if c.isdigit()])

            # Open the image
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                img_resized = img.resize(size)

                output_dir = os.path.join(output_folder, noun_part)
                os.makedirs(output_dir, exist_ok=True)

                # Save the resized image with just the number as the name
                output_image_path = os.path.join(output_dir, f"{number_part}.jpg")
                try:
                    img_resized.save(output_image_path)
                except:
                    print(f"Failed: {output_image_path}")
                    pass

                print(f"Saved: {output_image_path}")

 

if __name__ == "__main__":
    output_folder = 'testing'
    input_folder = 'HW7-Auxilliary/data/' + output_folder
    resize_and_save_images(input_folder, output_folder)