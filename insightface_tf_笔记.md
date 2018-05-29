# insightface_tf_笔记
## 基于项目
[Insightface_TF](https://github.com/auroua/InsightFace_TF)
- **介绍**：该项目作者将原版mxnet的InsightFace用TF重写
- **模型结构**：作者已经用TF代码将InsightFace的模型结构搭建完毕
- **训练数据**：作者用的就是mxnet原版项目中的训练数据，然后用脚本转换成TF格式
- **训练流程**：执行`train_nets.py`进行训练，在p2.xlarge(K80)上若要达到99.6%准确度，花费约两周；在g3.4large(M60)花费约一周
## 所需环境
首先确认机子上CUDA的版本，使用`nvcc --version`来看版本，应该是8.0或者是9.0  
根据版本，使用`pip3 install mxnet-cu80` 或者 `pip3 install mxnet-cu90`  
`pip3 install tensorlayer` 如果在执行训练脚本时报错，尝试`pip3 install tensorlayer==1.7`
## 准备数据
- 训练已经传到S3上，直接`aws s3 cp s3://kiwi-ai-ruobo/Insightface-tf-train/tfrecords/tran.tfrecords /YOUR_TRAIN_PATH/`即可
- LFW测试数据已经传到S3上，直接`aws s3 cp s3://kiwi-ai-ruobo/Insightface-tf-train/faces_ms1m_112x112 /YOUR_LFW_PATH/ --recursive`即可
## 编辑&运行训练脚本
开一个screen/session
``` bash
git clone https://github.com/ruobop/InsightFace_TF
cd InsightFace_TF
python3 train_nets.py --eval_db_path=/YOUR_LFW_PATH/ --tfrecords_file_path=/YOUR_TRAIN_PATH/
```
你可以调整`--batch_size`参数，默认是`--batch_size=48`，根据显存大小来调整  
另开一个screen/session
``` bash
tensorboard --logdir=./output/summary/
```
远程浏览器访问`IP_ADDRESS:6006`即可
## 查看当前训练LFW结果
下载log文件到本地
``` bash
scp -i kiwi-ai-wuxiang.pem ubuntu@YOUR_IP_ADDRESS:insightface_tf/output/logs/LOG_FILE_NAME LOCAL_LOG_PATH
```
在本地编辑如下python脚本，例如保存为parse.py
``` python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description='Parse a log file.')
parser.add_argument('log_path', help='path of the log file')
args = parser.parse_args()

path = args.log_path

f = open(path, 'r')
lines = f.readlines()
f.close()

if 'Best' in lines[-3]:
    iter_line = lines[-5]
    acc_line = lines[-4]
else:
    iter_line = lines[-2]
    acc_line = lines[-1]

#print(iter_line)
#print(acc_line)

iter_list = iter_line.split('\n')[0].split(',')
acc_list = acc_line.split('\n')[0].split(',')

adict = {}
for i in range(len(iter_list)):
    adict[int(iter_list[i])] = float(acc_list[i])

sorted_list = sorted(adict.items(), key=lambda d: d[0])
for sl in sorted_list:
    print('iter=%d, lfw_acc=%.4f' % (sl[0], sl[1]))

sorted_list_by_val = sorted(adict.items(), key=lambda d: d[1])
sl = sorted_list_by_val[-1]
print('###########################\nBest result: iter=%d, lfw_acc=%.4f' % (sl[0], sl[1]))
```
在本地运行脚本
``` bash
python3 parse.py LOCAL_LOG_PATH
```

