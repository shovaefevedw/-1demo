# PhotoDepth 3D

把一张普通照片变成带颜色的 2.5D 点云。深度来自 Depth Anything V2 的相对视差，不是以米为单位的真实距离。程序只为照片里看得到的表面生成点，不会补出被挡住的几何。

适合街景、房屋、树木、道路和风景。点云可以在浏览器里旋转查看，也可以导出成 PLY，供以后的动画或粒子使用。

## 环境

- macOS，Apple Silicon（M1 及更新）
- Python 3.11
- Node.js 20 或更新版本（本机用 Node 自带的 npm 即可）
- 不需要 CUDA、NVIDIA、Docker 或付费 API

有 Apple Metal 时，PyTorch 会使用 MPS。没有 MPS 时自动改用 CPU。

## 安装

在项目目录中：

```bash
chmod +x scripts/*.sh
./scripts/setup_backend.sh
./scripts/setup_frontend.sh
```

`setup_backend.sh` 会寻找 `python3.11`。如果命令不在 PATH 里，它也会使用 Homebrew 的 `/opt/homebrew/bin/python3.11`。没有 Python 3.11 时：

```bash
brew install python@3.11
```

Open3D 在 Apple Silicon 上还需要 Homebrew 的 libusb。安装脚本会在导入失败时自动安装它；也可以事先执行 `brew install libusb`。

## 模型权重

使用 Depth Anything 作者发布在 Hugging Face 上的官方 V2 模型：

| 界面选项 | 模型 | 适合看什么 |
| --- | --- | --- |
| V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` | 整体体积，最快 |
| V2 Base | `depth-anything/Depth-Anything-V2-Base-hf` | 更清楚的形体 |
| V2 Large | `depth-anything/Depth-Anything-V2-Large-hf` | 这组里最清楚的形体，头和喙更完整 |

默认用 V2 Small。要看清主体轮廓时选 V2 Large。第一次点击 Generate Depth 时会下载对应权重，缓存在 `backend/models/hf/`。这个目录不会进 Git。

Depth Pro 和 Cooperate 已从界面移除。在这台机器上，Depth Pro 的深度不跟着照片走，混进结果后会把原来的形体抹成另一张图。

下载失败时程序会报错，不会用假的深度图代替。

## 启动

```bash
./scripts/dev.sh
```

然后打开 [http://127.0.0.1:5173](http://127.0.0.1:5173)。后端在 `127.0.0.1:8000`。

## 使用

1. 把 JPG、JPEG 或 PNG 拖进左侧，或点击选择文件。原图会立刻显示。
2. 选择模型。日常使用 Small。
3. 点击 **Generate Depth**。这一步才会运行 AI。换照片或更换模型后需要再点一次。
4. 点击 **Generate Point Cloud**。中间出现一张平面的点云照片，不是可以旋转的三维模型。近处的点更大、更亮，远处的点更小、更暗、更密。
5. 右侧滑杆会立刻重画这张照片，不会重新跑模型。Dot Contrast 拉大近处大圆点和远处小圆点的差别。Dot Color 改圆点颜色，Background 改背景。圆点是圆的，近处更大更亮，远处更小更暗，默认密度大约是之前的三到四倍。
6. 导出：
   - **Download 16-bit Depth PNG**：单通道 uint16 相对视差，文件名 `depth_16bit.png`
   - **Download Point Photo**：平面点云照片 `pointcloud.png`。点的位置还在原图像素上，大小和深浅表示远近
   - **Download PLY**：带 XYZ 和黑白深浅的 `pointcloud.ply`。Open3D 写出的坐标是 double，颜色是 uchar，CloudCompare 和 Blender 可以打开
   - **Download XYZ**：ASCII `x y z r g b`
   - **Download All Results**：上面的文件，加上 `depth_preview.png`、`metadata.json` 和 `project_settings.json`

`depth_preview.png` 只是 8 位彩色可视化。数值越大（预览里越亮）表示模型认为越近。它不是原始深度。

## 深度和坐标

Depth Anything V2 的相对模型输出的是视差：数值越大，表面越近。16 位 PNG 按这个顺序做最小–最大归一化，映射到 0–65535。无效值不参与范围计算，并记为 0。`metadata.json` 会记下模型、原图尺寸、原始预测的最小和最大值、归一化方法和方向。

点云里的 Z 是虚拟场景单位，不是米：

```text
closeness = 将视差按最小–最大归一化到 0–1（1 表示更近）
若打开 Invert Depth，则 closeness = 1 - closeness
Z = near + (1 - closeness) * (far - near) * depth_strength
X = (u - cx) * Z / fx
Y = -(v - cy) * Z / fy
```

最后看到的点云是一张二维照片。每个点仍落在原图的像素位置上，不会把画面拉成立体模型。近处的点更大、更亮、更疏，远处的点更小、更暗、更密，体积靠大小和深浅来看。点的颜色与照片无关。PLY 里仍保留虚拟的针孔坐标，方便需要坐标的软件打开，但页面上展示的是这张平面照片。

Generate Depth 会把模型输出整理成前后顺序：物体内部的细碎起伏被压平，物体边缘不再是断层，而是一段朝向或远离相机的渐变。16 位 PNG 保存的就是这张整理后的相对视差，数值越大仍然表示越近。

如果模型输出的深度比照片小，会用双三次插值放大到照片分辨率，并限制在模型原始数值范围内。这只是对齐分辨率，不会制造真实的高频几何。Depth Smoothing 大于 0 时，只对深度图再做一次模糊，不读取照片颜色。

预览默认最多 50 万个点。Full HD 原图和完整深度会保留，把 Maximum Points 调高后可以导出更密的云。1080p 全像素大约 200 万点，上限是 300 万。

## 测试

```bash
./scripts/test.sh
```

集成测试会用 `sample_images/street.jpg` 真正运行 Small 模型。这张街景照片来自 Wikimedia Commons，作者 Rene Hernandez，许可为 CC BY-SA 2.0，说明见 `sample_images/SOURCE.txt`。第一次运行需要网络以下载权重。

## 已知限制

- 深度是相对视差，不能当成测量距离。
- 这是 2.5D。物体背后、玻璃内部和被挡住的结构不会出现。
- 深度跳变的边缘上，相邻像素可能拉出一条薄的过渡，这是单张照片投影的常见现象。
- 放大深度图不会增加真实几何细节。
- Large 模型可能因内存不足失败。这时换 Small，或换一张更小的照片。
- 照片长边很大时，单张上限约为 1200 万像素、30 MB。

## 以后可以加的部分

当前流水线是：读取照片、估计深度、归一化、按虚拟相机投影、导出。分割、多图、视频、点云动画、群集运动、高斯泼溅和长曝光效果都可以包在这条流水线外面，不需要改掉现有的单张照片路径。第一版没有实现这些功能。
