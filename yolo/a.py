import cv2
import os
import argparse

def crop_detected_objects(image_folder, label_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            image = cv2.imread(image_path)
            if image is None:
                continue
            h, w, _ = image.shape
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    for i, line in enumerate(lines):
                        elements = line.split()
                        if len(elements) < 5:
                            continue
                        class_id, x_center, y_center, width, height = map(float, elements)
                        x_center *= w
                        y_center *= h
                        width *= w
                        height *= h
                        x1 = max(0, int(x_center - width / 2))
                        y1 = max(0, int(y_center - height / 2))
                        x2 = min(w, int(x_center + width / 2))
                        y2 = min(h, int(y_center + height / 2))
                        cropped_image = image[y1:y2, x1:x2]
                        output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_crop_{i}.jpg")
                        cv2.imwrite(output_path, cropped_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop detected objects from images using YOLOv5 annotations.")
    parser.add_argument('image_folder', type=str, help='Path to the folder containing images.')
    parser.add_argument('label_folder', type=str, help='Path to the folder containing YOLO annotation files.')
    parser.add_argument('output_folder', type=str, help='Path to the folder where cropped images will be saved.')

    args = parser.parse_args()
    crop_detected_objects(args.image_folder, args.label_folder, args.output_folder)
