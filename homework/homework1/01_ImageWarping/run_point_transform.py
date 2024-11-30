import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    method = "MLS"
    #method = "RBF"
    warped_image = np.array(image)
    print("image.shape:", image.shape)
    ### FILL: 基于MLS or RBF 实现 image warping
    h, w = warped_image.shape[:2]
    n = source_pts.shape[0]
#示例：简单的一个左右镜面反转映射
    '''u = np.tile(np.arange(0, h), (w, 1)).T
    v = np.flip(np.tile(np.arange(0, w), (h, 1)),axis=1)
    u = np.float32(u)
    v = np.float32(v)
    warped_image = cv2.remap(warped_image,v,u,interpolation=cv2.INTER_LINEAR)'''
#算法
    #image：h*w*3
    #source_pts：n*2
#默认为恒等映射，U为x的映射矩阵，V为y的映射矩阵
    U = np.tile(np.arange(0, h), (w, 1)).T
    V = np.tile(np.arange(0, w), (h, 1))

    # MLS实现
    if method == "MLS":
        # Affine Deformations
        # 逆光流：反向映射，不会出现黑色线条
        # 输入的控制点坐标为(y,x)形式，因此要做个交换
        [source_pts, target_pts] = [target_pts[:,[1,0]], source_pts[:,[1,0]]]
        #构造w_i的矩阵
        for i in range(h):
            for j in range(w):
                v = np.array([i, j])# 1*2
                # W[i]=1 / (|source_pts[i]-v|**(2*alpha)+eps)
                # 行向量1*n
                W_ij = 1 / (np.sum(((source_pts - v) ** 2), axis=1) ** alpha + eps)
                # 列向量n*1
                W_ijT = W_ij.reshape(-1, 1)
                sum_w = np.sum(W_ij)
                p_ = np.sum(W_ijT * source_pts, axis=0) / sum_w
                q_ = np.sum(W_ijT * target_pts, axis=0) / sum_w
                source_pts_ = source_pts - p_
                target_pts_ = target_pts - q_
                # pwp=\sum p_i^Tw_ip_i 2*2矩阵
                #pwp = np.sum(W_ij * np.sum(source_pts_ ** 2, axis=1))
                pwp = W_ij * source_pts_.T @ source_pts_
                uv = q_ + (v - p_) @ np.linalg.inv(pwp) @ (W_ij * source_pts_.T) @ target_pts_
                #x = int(min(max(0, uv[0]), h-1))
                #y = int(min(max(0, uv[1]), w-1))
                x = int(uv[0])
                y = int(uv[1])
                #U[x][y] = i
                #V[x][y] = j
                U[i][j] = x
                V[i][j] = y

    # RBF实现：
    if method == "RBF":
        lamta = 0#(10e-04*w*h)**1.5*n
        sigma = (h+w)/12*0.2
        sigma2 = sigma**2
        A = np.ones((n + 3, n + 3))
        I = np.eye(n)
        # 仍然用逆光流
        p_XY = np.array(target_pts[:, [1,0]])
        q_XY = np.array(source_pts[:, [1,0]])
        g = np.array([[np.sum((p_XY[i]-p_XY[j])**2)
               for j in range(n)]
             for i in range(n)])
        G = np.exp(-g / sigma2)
        A[:n, 3:] = G - lamta * I
        A[:n, 1:3] = p_XY
        A[n+1:n+3, 3:] = p_XY.T
        A[n:n+3, :3] = 0
        xy = np.vstack((q_XY, np.array([[0, 0], [0, 0], [0, 0]])))
        ab, res, rank, s = np.linalg.lstsq(A, xy, rcond=None)
        for i in range(h):
            for j in range(w):
                v = np.array([i, j])
                xyg = [1, i, j]
                xyg = xyg + [np.exp(-np.sum((v-p_XY[k])**2)/sigma2) for k in range(n)]
                xyg = np.array(xyg)
                uv = xyg @ ab
                x = int(uv[0])
                y = int(uv[1])
                U[i][j] = x
                V[i][j] = y
    warped_image = cv2.remap(warped_image, np.float32(V), np.float32(U), interpolation=cv2.INTER_LINEAR)
    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
