import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from core_engine import color_match_with_histogram
import tkinter as tk
from tkinter import filedialog



# 用接收输入并防止输错（比如不小心打成了字母）
def get_param(prompt_text, default_value):
    user_input = input(f"{prompt_text} (直接回车默认 {default_value}): ").strip()
    if not user_input:
        return default_value
    try:
        return float(user_input)
    except ValueError:
        print(f"  [!] 输入无效，将使用默认值 {default_value}")
        return default_value

# 隐藏 Tkinter 主窗口
root = tk.Tk()
root.withdraw() 

print("=====================================")
print("      色彩与光影仿色工具      ")
print("=====================================")

try:
    print("\n[1/3] 请在弹出的窗口中选择【要处理的原图】(支持jpg，jpeg，png格式)...")
    SOURCE_IMAGE = filedialog.askopenfilename(title="选择要改变颜色的原图，支持jpg，jpeg，png格式", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not SOURCE_IMAGE:
        print("已取消选择，程序退出。")
        exit()

    print("\n[2/3] 请选择【提供色彩参考的目标图】(支持jpg，jpeg，png格式)...")
    TARGET_IMAGE = filedialog.askopenfilename(title="选择提供参考的目标照片, 支持jpg，jpeg，png格式",filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not TARGET_IMAGE:
        print("已取消选择，程序退出。")
        exit()

    print("\n[3/3] 请设置调色参数 (0.0 到 1.0 之间)：")
    # 动态获取你的参数
    p_blend = get_param("  ▶ 融合强度 [控制色彩向目标靠拢程度]", 0.5)
    p_sim   = get_param("  ▶ 保色阈值 [控制色相相似保护程度]", 0.5)
    p_hist  = get_param("  ▶ 光影强度 [控制明度直方图向目标靠拢程度]", 0)

    # 自动生成输出文件名
    base_name, ext = os.path.splitext(SOURCE_IMAGE)
    OUTPUT_IMAGE = f"{base_name}_仿色结果.png"

    print("\n开始处理，请稍候...")
    start_time = time.time()
    
    # 把输入的参数传给核心算法
    color_match_with_histogram(
        SOURCE_IMAGE, 
        TARGET_IMAGE, 
        OUTPUT_IMAGE, 
        blend_strength=p_blend, 
        similarity_threshold=p_sim, 
        hist_strength=p_hist
    )
    
    print(f"\n✅ 全部完成！耗时: {time.time() - start_time:.2f} 秒")
    print(f"📁 结果已保存至: {OUTPUT_IMAGE}")
    
except Exception as e:
    print(f"\n❌ 运行出错: {e}")

input("\n按回车键退出程序...")