import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision.transforms.functional as F
from PIL import Image
import math
import time
from color_utils import rgb_to_hsv,hsv_to_rgb,match_histogram

def color_match_with_histogram(source_path, target_path, output_path, blend_strength, similarity_threshold, hist_strength):
    """
    参数：source_path：原始图，target_path：目标图，blend_strength：融合强度，similarity_threshold：色彩保护阈值， hist_strength: 直方图匹配强度。

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} | 融合强度: {blend_strength} | 保色阈值: {similarity_threshold} | 光影匹配强度: {hist_strength}")

    source_pil = Image.open(source_path).convert('RGB')
    target_pil = Image.open(target_path).convert('RGB')

    source_tensor = F.to_tensor(source_pil).to(device)
    target_tensor = F.to_tensor(target_pil).to(device)

    # --- 1. 基础灰度的色彩映射---
    weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(3, 1, 1)#NTSC 灰度转换系数
    source_gray = (torch.sum(source_tensor * weights, dim=0) * 255).round().long()
    target_gray = (torch.sum(target_tensor * weights, dim=0) * 255).round().long()

    target_rgb_flat = target_tensor.view(3, -1)#把图片转成矩阵，每一列就是一个像素的RGB值
    target_gray_flat = target_gray.view(-1)#把灰度数据转成一维数组

    sum_rgb = torch.zeros(3, 256, device=device)
    sum_rgb[0].scatter_add_(0, target_gray_flat, target_rgb_flat[0])
    sum_rgb[1].scatter_add_(0, target_gray_flat, target_rgb_flat[1])
    sum_rgb[2].scatter_add_(0, target_gray_flat, target_rgb_flat[2])#建立灰度-色彩查找表，一个灰度就对应一组RGB数值

    counts = torch.bincount(target_gray_flat, minlength=256).float()#计算sum_rgb中的数值是多少个像素得到的
    valid_mask = counts > 0
    avg_rgb = torch.zeros(3, 256, device=device)
    for c in range(3):
        avg_rgb[c][valid_mask] = sum_rgb[c][valid_mask] / counts[valid_mask]#算平均值，某个灰度中的RGB数值的平均值

    valid_indices = torch.where(valid_mask)[0]#单参数的torch.where()返回valid_mask中True的位置索引
    for g in range(256):
        if not valid_mask[g]:   #如果g是没颜色的
            distances = torch.abs(valid_indices - g)  #那就看看g旁边的有颜色的像素
            nearest_idx = valid_indices[torch.argmin(distances)] #取离g最近的有颜色的
            avg_rgb[:, g] = avg_rgb[:, nearest_idx]#把那个颜色给g

    mapped_rgb_flat = avg_rgb[:, source_gray.view(-1)]#匹配颜色
    mapped_tensor = mapped_rgb_flat.view(3, source_gray.shape[0], source_gray.shape[1])#转回3维张量
    
# --- 2. 双分支色彩融合 (保色逻辑) ---
    
    # 【提前执行转换】先把原图和映射图转为 HSV，供后续计算距离和混合使用
    source_hsv = rgb_to_hsv(source_tensor)
    mapped_hsv = rgb_to_hsv(mapped_tensor)
    
    # 1. 提取色相 (H 通道)
    source_h = source_hsv[0:1]
    mapped_h = mapped_hsv[0:1]
    
    # 2. 计算环形色相距离并转为相似度
    h_diff = torch.abs(source_h - mapped_h)
    circular_h_distance = torch.minimum(h_diff, 1.0 - h_diff)
    similarity = 1.0 - (circular_h_distance / 0.5)
    
    # 3. 创建判定分流的遮罩
    safe_threshold = torch.clamp(torch.tensor(similarity_threshold), 0.0, 1.0)
    high_match_mask = similarity >= 1-safe_threshold
    
    # 分支 A：匹配度达标，正常混合 RGB
    branch_a_rgb = (source_tensor * (1.0 - blend_strength)) + (mapped_tensor * blend_strength)
    
    # 分支 B：匹配度不达标，锁死原图色相 (H)，混合饱和度 (S) 与明度 (V)
    new_s = (source_hsv[1:2] * (1.0 - blend_strength)) + (mapped_hsv[1:2] * blend_strength)
    new_v = (source_hsv[2:3] * (1.0 - blend_strength)) + (mapped_hsv[2:3] * blend_strength)
    branch_b_hsv = torch.cat([source_hsv[0:1], new_s, new_v], dim=0)
    branch_b_rgb = hsv_to_rgb(branch_b_hsv)
    
    # 4.合并分支
    output_tensor = torch.where(high_match_mask.expand_as(source_tensor), branch_a_rgb, branch_b_rgb)
    # ------------------------------------------------------

    output_hsv = rgb_to_hsv(output_tensor)
    #为那些颜色很淡的像素创建一个蒙版
    mask_pale = (source_hsv[1:2] < 5.0 / 255.0) & (source_hsv[2:3] > 0.8) #找出那些颜色很淡的像素
    mask_pale_3d = mask_pale.expand_as(source_tensor)
    protected_tensor = torch.where(mask_pale_3d, source_tensor, output_tensor)#这里是torch.where的三参数运算，mask_pale_3d中为True取source_tensor的值，False就取output_tensor的值，最后返回一个三维张量
    protected_hsv = rgb_to_hsv(protected_tensor)
    target_hsv = rgb_to_hsv(target_tensor)
    protected_v = protected_hsv[2:3]
    target_v = target_hsv[2:3]
    # =======================================================
    # --- 带强度控制的明度直方图匹配 ---
    # =======================================================
    
    
    output_v = output_hsv[2:3]
    target_v = target_hsv[2:3]
    
    # 计算出如果 100% 匹配应该是什么样
    matched_v = match_histogram(protected_v, target_v)
    
    # 核心修改：按用户设置的阈值 (hist_strength) 进行加权混合
    hist_strength = max(0.0, min(1.0, float(hist_strength)))
    final_v = (protected_v * (1.0 - hist_strength)) + (matched_v * hist_strength)
    
    # 拼接最终的 HSV 并转回 RGB
    final_hsv = torch.cat([protected_hsv[0:2], final_v], dim=0)
    final_output_tensor = hsv_to_rgb(final_hsv)
    # =======================================================

    # 4. 限制数值并保存 (不变)
    final_output_tensor = final_output_tensor.clamp(0, 1)
    output_pil = F.to_pil_image(final_output_tensor.cpu())
    
    ext = output_path.split('.')[-1].lower()
    if ext in ['jpg', 'jpeg']:
        output_pil.save(output_path, quality=100, subsampling=0)
    else:
        output_pil.save(output_path)
        
    print(f"调色完成！已保存至: {output_path}")
    return final_output_tensor