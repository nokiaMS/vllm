# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from PIL import Image  # 导入PIL图像处理库


def rescale_image_size(
    image: Image.Image, size_factor: float, transpose: int = -1
) -> Image.Image:
    """Rescale the dimensions of an image by a constant factor.
    按常数因子缩放图像尺寸。
    """
    new_width = int(image.width * size_factor)  # 计算新的宽度
    new_height = int(image.height * size_factor)  # 计算新的高度
    image = image.resize((new_width, new_height))  # 调整图像大小
    if transpose >= 0:  # 如果指定了转置参数
        image = image.transpose(Image.Transpose(transpose))  # 执行图像转置操作
    return image  # 返回缩放后的图像


def rgba_to_rgb(
    image: Image.Image,
    background_color: tuple[int, int, int] | list[int] = (255, 255, 255),
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color.
    将RGBA图像转换为RGB格式，并填充背景颜色。
    """
    assert image.mode == "RGBA"  # 断言图像模式为RGBA
    converted = Image.new("RGB", image.size, background_color)  # 创建带背景色的新RGB图像
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel  # 使用alpha通道作为蒙版粘贴图像
    return converted  # 返回转换后的图像


def convert_image_mode(image: Image.Image, to_mode: str):
    """将图像转换为指定的颜色模式。"""
    if image.mode == to_mode:  # 如果已经是目标模式
        return image  # 直接返回
    elif image.mode == "RGBA" and to_mode == "RGB":  # 如果是RGBA转RGB
        return rgba_to_rgb(image)  # 使用专门的RGBA转RGB函数
    else:  # 其他模式转换
        return image.convert(to_mode)  # 使用PIL内置的模式转换
