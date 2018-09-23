FashionAI关键点定位比赛网址[传送门](https://tianchi.aliyun.com/competition/information.htm?spm=5176.11165320.5678.2.73a22af1h2wKGc&raceId=231648)
## 1.基本思路
本次FashionAI关键点定位比赛我们采取的基本思路和方法是先利用提供的关键点标注文件生成
包围关键点的bounding box标注文件，并制作成COCO json格式，利用目标检测框架detectron
中的Faster R-CNN网络为五类服饰训练检测模型，得到服饰的bounding box后再利用回归模型
在bounding box内对关键点进行回归。后面因关键点定位效果不够理想，转而用Mask R-CNN直接
训练服饰关键点定位模型，而又由于skirt类的Mask R-CNN效果不够理想，对skirt类的关键点
定位用了CPM模型进行替代。

## 2.执行依赖的环境和库
Ubuntu16.04+CUDA8.0+cudnn5.1+Python2.7(包括opencv2和numpy等其它一些常用python库）
+Caffe2+detectron

## 3.训练步骤
**1. 数据处理**

使用preprocess文件夹下的make_bbox_coco_annotation.py文件中的相关函数生成COCO json
格式的bounding box标注文件和关键点标注文件，并按照detectron的要求在
detectron/lib/datasets/data目录下添加训练和测试数据集文件夹（使用的是软连接），
并在datasets目录下的dataset_catalog.py文件中进行注册

**2. 训练单类服饰的目标检测模型**

1)在detectron/configs/getting_started/FashionAI_bbox.yaml配置文件中对单类服饰的训练
参数进行配置，文件中以blouse的训练为例，其它类服饰只需修改Train和Test字段的数据集元
组以及模型输出路径OUTPUT_DIR即可，其它参数保持不变

2)在detectron/tools目录下执行```python train_net.py --cfg ../configs/getting_started/FashionAI_bbox.yaml```
命令即可开始训练

3)训练结束后，程序默认会自动调用测试函数对训练得到的最终模型进行测试，也可手动执行测试
程序，是在detectron/tools目录下执行```python test_net.py --cfg ../configs/getting_started/FashionAI_bbox.yaml --wts 训练好的模型权值文件路径```

4)执行预测需要使用detectron/tools目录下的infer_simple.py中的write_infer_bbox函数，执行
预测的脚本命令格式如下
```python infer_simple.py --cfg ../configs/getting_started/FashionAI_bbox.yaml --wts 训练好的模型权值文件路径 --output-dir 预测结果的输出路径 --input-data 需要预测的.csv 文件需要预测的图片路径```

**3. 训练单类服饰的关键点检测模型**

1)在detectron/configs/getting_started/FashionAI_keypoint.yaml配置文件中对单类服饰的训练
参数进行配置，文件中以blouse的训练为例，其它类服饰只需修改Train和Test字段的DATASETS数据集元
组、TRAIN字段的WEIGHTS(用前面训练好的对应服饰的目标检测模型）、KRCNN字段的NUM_KEYPOINTS以及
模型输出路径OUTPUT_DIR即可，其它参数保持不变

2)由于Mask R-CNN默认支持的只是人体关键点检测，若要训练服饰关键点,还需要对
detectron/lib/datasets/json_dataset.py文件中的self.keypoint_flip_map字典进行修改，
换成对应服饰类的关键点映射字典；此外，还需要对detectron/utils/keypoints.py文件也做
类似的修改，把get_keypoints函数中的keypoints和keypoint_flip_map换成对应服饰类的。

3)在detectron/tools目录下执行python train_net.py --cfg ../configs/getting_started/FashionAI_keypoint.yaml命令即可开始训练

4)训练结束后，程序默认会自动调用测试函数对训练得到的最终模型进行测试，也可手动执行测试
程序，是在detectron/tools目录下执行```python test_net.py --cfg ../configs/getting_started/FashionAI_keypoint.yaml --wts 训练好的模型权值文件路径```

5)执行预测需要使用detectron/tools目录下的infer_simple.py中的write_infer_kpts函数，执行
预测的脚本命令格式如下
```python infer_simple.py --cfg ../configs/getting_started/FashionAI_keypoint.yaml --wts 训练好的模型权值文件路径 --output-dir 预测结果的输出路径 --input-data 需要预测的.csv文件 需要预测的图片路径```

**4. visualize.py文件中含有一些可视化模型训练过程和结果的函数**
