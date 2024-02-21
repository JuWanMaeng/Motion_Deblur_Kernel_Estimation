from PIL import Image

def crop_image(image_path, output_path, crop_rectangle):
    """
    Crops a rectangle from an image and saves the cropped image.

    :param image_path: Path to the input image.
    :param output_path: Path to save the cropped image.
    :param crop_rectangle: A tuple of (x1, y1, x2, y2) coordinates.
    """
    with Image.open(image_path) as img:
        cropped_image = img.crop(crop_rectangle)
        cropped_image.save(output_path)

# 사용 예시
image_path = 'results/NAF/wall/recon/img_walls.png'  # 대상 이미지 경로
output_path = 'crop.png'  # 자른 이미지 저장 경로
crop_rectangle = (134,160,359,400)  # 자르고 싶은 영역의 좌표 (x1, y1, x2, y2)

crop_image(image_path, output_path, crop_rectangle)

