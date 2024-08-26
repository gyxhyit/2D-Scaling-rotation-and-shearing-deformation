import cv2
import numpy as np
import matplotlib.pyplot as plt

def transform_image(image_path, shear_factor, scale_factor, rotation_angle):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return None, None

    # 获取图像的高度和宽度
    (height, width) = image.shape[:2]

    # 1. 缩小图像
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)
    scaled_image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

    # 2. 计算旋转矩阵
    center = (scaled_width // 2, scaled_height // 2)
    M_rotation = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # 计算旋转后的图像尺寸
    abs_cos = abs(M_rotation[0, 0])
    abs_sin = abs(M_rotation[0, 1])
    new_width = int(scaled_height * abs_sin + scaled_width * abs_cos)
    new_height = int(scaled_height * abs_cos + scaled_width * abs_sin)

    # 更新旋转矩阵以考虑图像中心的偏移
    M_rotation[0, 2] += (new_width / 2) - center[0]
    M_rotation[1, 2] += (new_height / 2) - center[1]

    # 应用旋转变换，背景色设置为白色
    rotated_image = cv2.warpAffine(scaled_image, M_rotation, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # 3. 计算仿射变换矩阵，使图像变成上边窄、下边宽的梯形
    src_points = np.float32([[0, 0], [new_width - 1, 0], [0, new_height - 1], [new_width - 1, new_height - 1]])
    dst_points = np.float32([[new_width * shear_factor, 0], [new_width * (1 - shear_factor), 0], [0, new_height - 1], [new_width - 1, new_height - 1]])

    # 计算四点变换矩阵
    M_perspective = cv2.getPerspectiveTransform(src_points, dst_points)

    # 应用透视变换，背景色设置为白色
    sheared_image = cv2.warpPerspective(rotated_image, M_perspective, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return image, sheared_image

# 示例用法
image_path = 'C:/Users/admin/Desktop/321.jpg'
shear_factor = 0.2    # 梯形变换因子
scale_factor = 0.5    # 缩小因子
rotation_angle = 30   # 旋转角度（以度为单位）
original_image, transformed_image = transform_image(image_path, shear_factor, scale_factor, rotation_angle)

if original_image is not None and transformed_image is not None:
    # 展示原始图像和变换后的图像
    fig = plt.figure(figsize=(12, 6))

    # 原始图像
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax1.set_title('')
    ax1.axis('off')

    # 变换后的图像
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    ax2.set_title('')
    ax2.axis('off')

    plt.show()
