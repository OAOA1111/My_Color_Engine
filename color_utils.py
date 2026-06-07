import torch
# 色彩空间转换 
def rgb_to_hsv(rgb):
    cmax, _ = torch.max(rgb, dim=0, keepdim=True)
    cmin, _ = torch.min(rgb, dim=0, keepdim=True)
    '''
    取最大和最小值，torch在取张量的值时同时会生成索引，把索引赋值给_,相当于丢弃
    '''
    delta = cmax - cmin

    h = torch.zeros_like(cmax)
    s = torch.zeros_like(cmax) #初始化，生成一个形状和cmax一样的0矩阵
    v = cmax

    mask_s = cmax > 0#生成bool掩码，cmax>0的位置为True，如果是纯黑色cmax=0，为False
    s[mask_s] = delta[mask_s] / cmax[mask_s]#只针对cmax不为0的地方求饱和度，为0的地方是黑色，饱和度就是0

    mask_r = (cmax == rgb[0:1]) & (delta > 0)
    mask_g = (cmax == rgb[1:2]) & (delta > 0)
    mask_b = (cmax == rgb[2:3]) & (delta > 0)#分别提取三个通道的数据
    '''
    cmax == rgb[0:1]：找出当前像素里，红色是不是最大的那个值。
    如果是，说明这个像素的基调是红色
    '''
    #处理色彩冲突
    mask_g = mask_g & (~mask_r)#在属于g的像素中去掉那些已经在r里面的
    mask_b = mask_b & (~mask_r) & (~mask_g)
    '''
    优先级判定 红 > 绿 > 蓝，
    '''
    #计算色相
    h[mask_r] = (rgb[1:2][mask_r] - rgb[2:3][mask_r]) / delta[mask_r]
    h[mask_g] = 2.0 + (rgb[2:3][mask_g] - rgb[0:1][mask_g]) / delta[mask_g]
    h[mask_b] = 4.0 + (rgb[0:1][mask_b] - rgb[1:2][mask_b]) / delta[mask_b]

    h = (h / 6.0) % 1.0 #把整个色相环设置在 0.0 到 1.0 的百分比区间
    return torch.cat([h, s, v], dim=0)#返回一个[3, Height, Width] 的 HSV 张量

def hsv_to_rgb(hsv):
    h, s, v = hsv[0:1], hsv[1:2], hsv[2:3] #重新把上面的转成hsv的张量分开
    i = (h * 6.0).floor()
    f = (h * 6.0) - i #将色调映射到 HSV 六边形模型的 6 个扇区：i 是扇区索引（0~5 的整数），f 是扇区内的小数偏移
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6.0#计算三个中间值，分别对应 RGB 转换中出现的三个亮度变体

    rgb = torch.zeros_like(hsv)#初始化张量，准备一张全是0的画布
    '''
    在6个色环扇区把rgb数值重新算出来
    '''
    '''
    每一个像素都有一个所在扇区的编码，计算某个mask的时候只计算bool值是Ture的像素
    '''
    mask0 = i == 0; rgb[0:1][mask0] = v[mask0]; rgb[1:2][mask0] = t[mask0]; rgb[2:3][mask0] = p[mask0]
    mask1 = i == 1; rgb[0:1][mask1] = q[mask1]; rgb[1:2][mask1] = v[mask1]; rgb[2:3][mask1] = p[mask1]
    mask2 = i == 2; rgb[0:1][mask2] = p[mask2]; rgb[1:2][mask2] = v[mask2]; rgb[2:3][mask2] = t[mask2]
    mask3 = i == 3; rgb[0:1][mask3] = p[mask3]; rgb[1:2][mask3] = q[mask3]; rgb[2:3][mask3] = v[mask3]
    mask4 = i == 4; rgb[0:1][mask4] = t[mask4]; rgb[1:2][mask4] = p[mask4]; rgb[2:3][mask4] = v[mask4]
    mask5 = i == 5; rgb[0:1][mask5] = v[mask5]; rgb[1:2][mask5] = p[mask5]; rgb[2:3][mask5] = q[mask5]
    '''
    以mask0为例，这是色相为0-60度的扇区，颜色为红到黄，那么r值就应该取v(最大亮度)，b值取p(最低亮度)，g取t(从暗变亮)
    这套算法最早由 Alvy Ray Smith 在 1978 年发明《Color Gamut Transform Pairs》
    '''
    return rgb
    



# ================= 工具函数：直方图匹配 =================
def match_histogram(source_v, target_v):
    """
    让 source_v (0.0-1.0) 的直方图分布完全匹配 target_v (0.0-1.0)
    """
    # 1. 放大到 0-255 离散整数区间以便统计直方图
    src_v_255 = (source_v * 255).round().long()
    tgt_v_255 = (target_v * 255).round().long()#把不同像素的明度×255再取整，这个操作在PS里就是直方图本来是个连续的曲线，然后变成离散的

    # 2. 统计明度分布 (Histogram)
    src_hist = torch.bincount(src_v_255.view(-1), minlength=256).float()
    tgt_hist = torch.bincount(tgt_v_255.view(-1), minlength=256).float()#用view函数和bincount统计函数把照片的明度转成PS里那种直方图的形式

    # 3. 计算累积分布函数 (CDF)
    src_cdf = torch.cumsum(src_hist, dim=0)
    tgt_cdf = torch.cumsum(tgt_hist, dim=0)#torch.cumsum累加，计算小于某个亮度的像素有多少

    # 4. 归一化 CDF 到 0.0 - 1.0 之间
    src_cdf = src_cdf / src_cdf[-1]
    tgt_cdf = tgt_cdf / tgt_cdf[-1]#把前面算的转成比例，例如 src_cdf[100] = 0.8，就说明在原图里，亮度为 100 的像素，比画面中 80% 的区域都要亮

    # 5. 构建查找表 (Look-Up Table)
    # 对于原图中的每一个亮度级别，去目标 CDF 中寻找最接近的百分比，取其亮度值作为新映射
    mapping = torch.zeros(256, dtype=torch.float32, device=source_v.device)#初始化
    for i in range(256):  
        diff = torch.abs(tgt_cdf - src_cdf[i]) # 计算目标图CDF与原图的差值
        mapping[i] = torch.argmin(diff).float() / 255.0#找差值最小的那个，torch.argmin函数返回的是当diff最小时，那个tgt_cdf对应的数

    # 6. 将原图明度按查找表进行替换
    matched_v = mapping[src_v_255]#把原图的百万个旧明度值（0-255的整数）直接当做“页码/指针”，去 mapping 字典里查出对应的新明度（小数），并保持原图的矩阵形状
    return matched_v