# insightface_tf_笔记
## 基于项目
[Insightface_TF](https://github.com/auroua/InsightFace_TF)
- **介绍**：该项目作者将原版mxnet的InsightFace用TF重写
- **模型结构**：作者已经用TF代码将InsightFace的模型结构搭建完毕
- **训练数据**：作者用的就是mxnet原版项目中的训练数据，然后用脚本转换成TF格式
- **训练流程**：执行`train_nets.py`进行训练，在p2.xlarge(K80)上若要达到99.6%准确度，花费约两周；在g3.4large(M60)花费约一周
## 准备数据
- 训练已经传到S3上，直接`aws s3 cp s3://kiwi-ai-ruobo/Insightface-tf-train/tfrecords/tran.tfrecords /YOUR_TRAIN_PATH/`即可
- LFW测试数据已经传到S3上，直接`aws s3 cp s3://kiwi-ai-ruobo/Insightface-tf-train/faces_ms1m_112x112 /YOUR_LFW_PATH/ --recursive`即可
## 编辑&运行训练脚本
开一个screen/session
``` sh
git clone https://github.com/ruobop/InsightFace_TF
cd InsightFace_TF
python3 train_nets.py --eval_db_path=/YOUR_LFW_PATH/ --tfrecords_file_path=/YOUR_TRAIN_PATH/
```
你可以调整`--batch_size`参数，默认是`--batch_size=48`，根据显存大小来调整
另开一个screen/session
``` sh
tensorboard --logdir=./output/summary/
```
远程连接`IP_ADDRESS:6006`即可
