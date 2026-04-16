# 开发指南 (Development Guide)

## 代码结构和规范

## 添加新功能

## 调试和测试

### 模型优化

参看[^rknn-toolkit2]的`doc/03_Rockchip_RKNPU_API_Reference_RKNN_Toolkit2_V2.3.2_CN.pdf`，这是官方的API使用文档。

把onnx模型转为rknn模型的极简脚本：

```python
# file name: onnx2rknn.py
from rknn.api import RKNN

rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model='tennis.onnx')
rknn.build(do_quantization=False)
rknn.export_rknn(export_path='tennis.rknn')
rknn.release()
```

### 连接主机

目前有adb, ssh, 调试串口三种连接方式。日常调试推荐用ssh连接。

用户名：`orangepi` 密码：`orangepi`

```
# 调试串口连接
minicom -D /dev/ttyUSB0 -b 1500000

# usb线adb连接
adb devices
adb shell

# ssh连接
sudo ip a add 192.168.1.2/24 dev <ethN>	# 设置主机的ip地址
ssh orangepi@192.168.1.10	# 开发板上靠近电源线的网口被设置为192.168.1.10
```

### 手动运行

系统启动后本程序会自动运行，可以手动运行观察输出。如下命令都是在开发板的系统上执行。

```
sudo systemctl stop chenlong-robot	# 停止本程序的运行，注意下次重启后本程序依然会自动运行
conda activate rknn			# 切换到本程序的运行环境
cd ~/Desktop-Wanderer	# 切换到本程序所在的目录
python -m src.main			# 手动运行本程序
```



### 附录

[^rknn-toolkit2]: https://github.com/airockchip/rknn-toolkit2.git 