# tf_pb_model_convert  

![logo](img/image.jpg)

Convert TF2.2 saved pb model directly  to TF1.13 and TF1.15 for inferencing  

将TF2.2中训练导出的模型pb文件转换为TF1中可推理的pb文件. 
------  

There are cases that researchers develop their deep learning model under the framework of TensorFlow2.2, while the product environment is Tensorflow1.13 or Tensorflow1.15. This code helps convert your TF2.2 SavedModel(using model.save()) into TF1 FrozenGraph pb file that can be loaded under TF1.13 / TF1.15 environment for inferencing.  

在企业应用中很多时候研发人员在TF2.2下训练的模型需要部署在TF1.13或TF1.15的生产环境下. 这个仓库将帮助你直接将TF2的SavedModel(用 model.save()保存)的代码转为TF1可推理的FrozenGraph的pb文件. 节约开发时间和环境差异导致的额外精力浪费. 

------  


## How to Use (Cn)  

参考博客: http://julianlu.com/blog/coding_pb_tf2_to_tf1/

或  

知乎文章: https://zhuanlan.zhihu.com/p/382652354

------  


## How to Use (En)

### Prelimiary   

Assume you have a trained model saved in TF2.2 using model.save() funciton, with the saved model directory architecture as:  
```shell
-tf22_model_save
    - saved_model.pb
    -assets/
    -variables/
```

### TF2 to TF1.15   

Step1: under tf2.2, covnert above saved tf2.2 model to tf1.15 compatiable pb file. Refer to  
```bash
tf2.2_SavedModel_to_tf1.15_pb.py
```
Above step result in a tf1.15 pb file, with model directory as: 
```bash
-tf115_pb_model
	-model.pb
```

Step2: under tf1.15, load above pb model for inference, refer to: 
```
infer_tf1.15_pb.py
```

[Done] 

### TF2 to TF1.15   

If we directly apply tf1.15 pb model under tf1.13, we will probably get error like:

[error](img/tf113_error.jpg)

In order to make the pb model from tf2.2 work under tf1.13, we need to do the following: 

Step1, apply steps in previous section to get a tf1.15 compatiable pb file from tf2.2.  

Step2: Under tf1.15, convert tf1.15 pb to tf1.15 (and tf1.13 compatibale) SavedModel, refer to: 
```bash
tf1.15_pb_to_SavedModel.py
```
above will result in an output model with directory structure as: 

```bash
-tf115_save_model
    -saved_model.pb
    variables/
```

Step3: Under tf1.13, load the tf1.15 SavedModel and convert it to tf1.13 pb model, refert to: 
```bash
tf1.13_SavedModel_to_pb.py
```

Now the conversion is done, you can make inference in tf1.13 using the converted pb file, refer to: 
```bash
infer_tf1.13_pb.py
```

Note that this tf1.13 pb model can be used directly in tf1.15 for inference, with the same inferencing code. 


[Enjoy !] 

