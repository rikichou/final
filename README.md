
# 0, 说在前面

有关该项目中所采用的策略和其他一些分析都会以开题报告的形式呈现，由于该报告的内容不适合在本notebook中显示，所以放在了同一个git仓库中的"project ayalysis.zip"压缩包中，如果需要了解的可以自行下载

# 1，数据集准备


数据集来自kaggle的dog vs cat主页（https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data）



```python
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import os
import h5py
import numpy as np

data_path = './'
train_dir_name = 'train'
test_dir_name = 'test'

resnet_50_model_save_name = 'model_resnet50.h5'
inceptionv3_model_save_name = 'model_inceptionv3.h5'
xception_model_save_name = 'model_xception.h5'

## check if the train and  test data is exist
if not isdir(data_path + train_dir_name):
    if not isfile(data_path + train_dir_name + '.zip'):
        print ("Please download train.zip from kaggle!")
        assert(False)
    else:
        with zipfile.ZipFile(data_path + train_dir_name + '.zip') as azip:
            print ("Now to extract %s " % (data_path + train_dir_name + '.zip'))
            azip.extractall()
    
if not isdir(data_path + test_dir_name):
    if not isfile(data_path + test_dir_name + '.zip'):
        print ("Please download test1.zip from kaggle!")
        assert(False)
    else:
        with zipfile.ZipFile(data_path + test_dir_name + '.zip') as azip:
            print ("Now to extract %s " % (data_path + test_dir_name + '.zip'))
            azip.extractall()
print ("Data is ready!")
```

    Data is ready!


简要地来观察一下样本数据


```python
train_dir = data_path + train_dir_name
test_dir = data_path + test_dir_name

## get all train filenames and test filenames
train_filenames = os.listdir(train_dir)
test_filenames = os.listdir(test_dir)

name_cats_clean = np.load("clean_cats_list.npy")
name_dogs_clean = np.load("clean_dogs_list.npy")

name_clean = list(name_cats_clean)+list(name_dogs_clean)
train_filenames = [x for x in train_filenames if x not in name_clean]
```


```python
import matplotlib.pyplot as plt
import cv2
import numpy as np
%matplotlib inline

dis_list = np.random.randint(len(train_filenames), size=(9))
dis_list = [train_filenames[index] for index in dis_list]

plt.figure(1, figsize=(13, 13))

for i,filename in enumerate(dis_list):
    image = cv2.imread('train/'+str(filename))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    ax1=plt.subplot(3,3,i+1)
    plt.imshow(image)
    plt.axis("off")
    plt.title(filename + "\n" + str(image.shape))
    
plt.show()
```


![png](output_5_0.png)


尝试观察了几次，发现样本的特点，猫狗基本都是照片的主体，并且非常清晰，有利于优化。并且都是500*500像素以下的。

## 1.1，将猫狗训练数据分开存放

为了尽快实现想法，我打算采用Keras平台，因为Keras已经包含了众多知名的模型，并且可以很轻易地加载与训练权重。而且Keras也提供了很方便的工具（image generator）来帮助我们直接从硬盘加载图片到模型，这样不但减少了代码量，而且利用Keras自带的程序能够极大减少内存的浪费（相比于手动一次性加载到内存）。但是image generator需要图片已经根据类别放入不同的文件夹。所以接下来的事情就是将训练和测试样本按照类别放入不同的文件夹。采用软链接的方式可以节约硬盘和时间。


```python
## get all dogs and cats
cat_names = filter(lambda x:x[:3] == 'cat', train_filenames)
dog_names = filter(lambda x:x[:3] == 'dog', train_filenames)
```

现在已经获得了文件名，现在需要另外建立一个文件夹，用于存储分开的猫狗图片。为了节省空间，这里采用建立软链接的方式来分开存储猫狗图片。


```python
## check if we did that
test_link_path = './test_link'
test_link_data = './test_link/data'

train_link_path = './train_link'
train_link_cat = './train_link/cat'
train_link_dog = './train_link/dog'

if not isdir(test_link_path):
    print ("Now to build %s!" % (test_link_path))
    os.makedirs(test_link_path)
    os.symlink('../' + test_dir_name, test_link_data)
    
if not isdir(train_link_path):
    print ("Now to build %s!" % (train_link_path))
    os.makedirs(train_link_cat)
    os.makedirs(train_link_dog)
    ## create link for the image
    for file in cat_names:
        os.symlink('../../' + train_dir_name+'/'+file, train_link_cat+'/'+file)
    for file in dog_names:
        os.symlink('../../' + train_dir_name+'/'+file, train_link_dog+'/'+file)

print ("Build all linkage complete!")
```

    Now to build ./train_link!
    Build all linkage complete!


# 2, 方案1--单个模型

首先我想到的第一个方案就是单个模型，在计算机视觉领域，有许多已经被证明了非常好用的模型，比如inception，resnet等，所以我接下来就是依次尝试各个模型，看看实际的效果怎么样？能否达到毕业项目的要求？并且我会预训练模型来进行训练，而不是从头训练，因为这些模型都已经在庞大的分类数据集中进行了训练，已经学习了足够的用于常用物体分类的‘知识’，因为图像的基本‘知识’是可以通用的，所以我采用迁移学习来对模型进行‘微调’，而不是从头学起。我打算采用inceptionv3,resnet-50,xception这三个模型来进行尝试。

## 2.1 resnet-50

初步采用的模型是ResNet-50，所以我们就需要按照该模型的要求来对图片进行预处理。
现在我们需要载入预训练的ResNet-50模型，并且，由于ResNet-50模型的输出是1000维的向量，而我们的功能是二分类，所以只需要输出单一的概率即可，所以需要替换掉原始的输出层换成我们的sigmoid激活函数，include_top=False就不会加载原模型的全连接层部分。


```python
from keras.applications import resnet50

## resNet-50 do not need preprocessing, so the resNet_input_shape is not neccessary
resNet_input_shape = (224,224,3)

res_x = Input(shape=resNet_input_shape)
res_x = Lambda(resnet50.preprocess_input)(res_x)
res_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=res_x, input_shape=resNet_input_shape)

print (res_model.output)
```

    Tensor("avg_pool/AvgPool:0", shape=(?, 1, 1, 2048), dtype=float32)


由于我们采用的预训练模型，暂时只是在ResNet-50上微调，不会对全连接层以外的其它层进行训练。所以如果每次都从上面的模型的输入端进行输入的话会产生很多重复的计算，显得不那么高效。那么这里有一个技巧，可以先把resnet-50模型的全连接层以前的输出向量（传说中的bottleneck features，以下均称为特征向量）预先计算并保存起来，由于只需要训练之后的全连接层，所以，将这些保存的特征向量作为样本，当做之后的全连接层的训练的输入就好了。


```python
vec_dir_name = "vect/"
resnet_50_vec_name = 'resnet-50.h5'
vec_dir_path = data_path + "vect"
resnet_50_vec_path = data_path + vec_dir_name + resnet_50_vec_name
print (vec_dir_path)
if not isdir(vec_dir_path):
    os.makedirs(vec_dir_path)
    print ("Make vector dir:%s" % (vec_dir_path))
    
"""
check if the resnet-50 vector file is exist
"""
if not isfile(resnet_50_vec_path):
    with h5py.File(resnet_50_vec_path, 'w') as f:
        print ("creating vector!!")
        out = GlobalAveragePooling2D()(res_model.output)
        res_vec_model = Model(inputs=res_model.input, outputs=out)
        
        ## save vector
        gen = ImageDataGenerator()
        test_gen = ImageDataGenerator()
        """
        classes = ['cat', 'dog'] -- cat is 0, dog is 1, so we need write this
        class_mode = None -- i will not use like 'fit_fitgenerator', so i do not need labels
        shuffle = False -- it is unneccssary
        batch_size = 64 
        """
        image_size = (224,224)
        train_generator = gen.flow_from_directory(train_link_path, image_size, color_mode='rgb', \
                                                  classes=['cat', 'dog'], class_mode=None, shuffle=False, batch_size=64)
        test_generator = test_gen.flow_from_directory(test_link_path, image_size, color_mode='rgb', \
                                                  class_mode=None, shuffle=False, batch_size=64)        
        """
        steps = None, by default, the steps = len(generator)
        """
        vector = res_vec_model.predict_generator(train_generator)
        test_vector = res_vec_model.predict_generator(test_generator)
        
        f.create_dataset('x_train', data=vector)
        f.create_dataset("y_train", data=train_generator.classes)
        f.create_dataset("test", data=test_vector)

```

    ./vect
    Make vector dir:./vect
    creating vector!!
    Found 24975 images belonging to 2 classes.
    Found 12500 images belonging to 1 classes.


现在直接读取保存的特征向量进行训练，参数如下，优化器采用Adam，学习速率采用默认的0.001，batch size为32，先训练20epoch看看结果。


```python
import numpy as np
from sklearn.utils import shuffle
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler
import math

with h5py.File(resnet_50_vec_path, 'r') as f:
    x_train = np.array(f['x_train'])
    y_train = np.array(f['y_train'])
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    
input_tensor = Input(shape=(2048,))
x = Dropout(0.4)(input_tensor)
x = Dense(1, activation='sigmoid', name='res_dense_1')(x)

res_top_model = Model(inputs=input_tensor, outputs=x)



#optimizer
#lr_base = 0.045
lr_base = 0.0002
decay_rate = 0.94
decay_steps = 2
#opt = RMSprop(lr=lr_base, rho=0.9, epsilon=1.0)
opt = RMSprop(lr=lr_base, rho=0.9)

# exponential rate decay:decayed_learning_rate = lr_base * decay_rate ^ (global_step / decay_steps)
def lr_scheduler(epoch):
    # calculate the new learning rate according to epoch number
    lr = lr_base * ((decay_rate)**math.floor(epoch/decay_steps))
    print ("Epoch %d , new lr==%f" % (epoch, lr))    
    
    return lr

scheduler = LearningRateScheduler(lr_scheduler)
res_top_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
hist = res_top_model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.2, callbacks=[scheduler])


#res_top_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#hist = res_top_model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.2)
```

    Train on 19980 samples, validate on 4995 samples
    Epoch 1/20
    Epoch 0 , new lr==0.000200
    19980/19980 [==============================] - 3s 137us/step - loss: 0.1458 - acc: 0.9439 - val_loss: 0.0450 - val_acc: 0.9848
    Epoch 2/20
    Epoch 1 , new lr==0.000200
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0491 - acc: 0.9827 - val_loss: 0.0353 - val_acc: 0.9882
    Epoch 3/20
    Epoch 2 , new lr==0.000188
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0430 - acc: 0.9851 - val_loss: 0.0333 - val_acc: 0.9892
    Epoch 4/20
    Epoch 3 , new lr==0.000188
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0395 - acc: 0.9854 - val_loss: 0.0319 - val_acc: 0.9896
    Epoch 5/20
    Epoch 4 , new lr==0.000177
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0367 - acc: 0.9878 - val_loss: 0.0319 - val_acc: 0.9888
    Epoch 6/20
    Epoch 5 , new lr==0.000177
    19980/19980 [==============================] - 2s 101us/step - loss: 0.0354 - acc: 0.9878 - val_loss: 0.0305 - val_acc: 0.9906
    Epoch 7/20
    Epoch 6 , new lr==0.000166
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0339 - acc: 0.9874 - val_loss: 0.0310 - val_acc: 0.9898
    Epoch 8/20
    Epoch 7 , new lr==0.000166
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0349 - acc: 0.9872 - val_loss: 0.0300 - val_acc: 0.9912
    Epoch 9/20
    Epoch 8 , new lr==0.000156
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0339 - acc: 0.9884 - val_loss: 0.0299 - val_acc: 0.9910
    Epoch 10/20
    Epoch 9 , new lr==0.000156
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0330 - acc: 0.9886 - val_loss: 0.0296 - val_acc: 0.9900
    Epoch 11/20
    Epoch 10 , new lr==0.000147
    19980/19980 [==============================] - 2s 101us/step - loss: 0.0330 - acc: 0.9881 - val_loss: 0.0294 - val_acc: 0.9898
    Epoch 12/20
    Epoch 11 , new lr==0.000147
    19980/19980 [==============================] - 2s 101us/step - loss: 0.0322 - acc: 0.9887 - val_loss: 0.0293 - val_acc: 0.9900
    Epoch 13/20
    Epoch 12 , new lr==0.000138
    19980/19980 [==============================] - 2s 101us/step - loss: 0.0332 - acc: 0.9886 - val_loss: 0.0301 - val_acc: 0.9906
    Epoch 14/20
    Epoch 13 , new lr==0.000138
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0336 - acc: 0.9887 - val_loss: 0.0289 - val_acc: 0.9902
    Epoch 15/20
    Epoch 14 , new lr==0.000130
    19980/19980 [==============================] - 2s 101us/step - loss: 0.0321 - acc: 0.9883 - val_loss: 0.0294 - val_acc: 0.9908
    Epoch 16/20
    Epoch 15 , new lr==0.000130
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0307 - acc: 0.9890 - val_loss: 0.0290 - val_acc: 0.9912
    Epoch 17/20
    Epoch 16 , new lr==0.000122
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0308 - acc: 0.9894 - val_loss: 0.0292 - val_acc: 0.9900
    Epoch 18/20
    Epoch 17 , new lr==0.000122
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0304 - acc: 0.9895 - val_loss: 0.0290 - val_acc: 0.9906
    Epoch 19/20
    Epoch 18 , new lr==0.000115
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0313 - acc: 0.9895 - val_loss: 0.0289 - val_acc: 0.9902
    Epoch 20/20
    Epoch 19 , new lr==0.000115
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0311 - acc: 0.9890 - val_loss: 0.0288 - val_acc: 0.9906



```python
hist = res_top_model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2, callbacks=[scheduler])
```

    Train on 19980 samples, validate on 4995 samples
    Epoch 1/5
    Epoch 0 , new lr==0.000200
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0314 - acc: 0.9895 - val_loss: 0.0298 - val_acc: 0.9906
    Epoch 2/5
    Epoch 1 , new lr==0.000200
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0311 - acc: 0.9896 - val_loss: 0.0286 - val_acc: 0.9908
    Epoch 3/5
    Epoch 2 , new lr==0.000188
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0302 - acc: 0.9898 - val_loss: 0.0291 - val_acc: 0.9900
    Epoch 4/5
    Epoch 3 , new lr==0.000188
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0287 - acc: 0.9896 - val_loss: 0.0284 - val_acc: 0.9906
    Epoch 5/5
    Epoch 4 , new lr==0.000177
    19980/19980 [==============================] - 2s 100us/step - loss: 0.0291 - acc: 0.9895 - val_loss: 0.0284 - val_acc: 0.9910


上面的结果看起来还不错，待会儿再在测试集上得出结果提交到kaggle。其实在得到这个结果之前犯了一个错误，就是从文件中读取出来x_train和y_train之后，没有对样本进行shuffle。导致验证集的loss在0.2到0.7不断地跳动，我刚开始还以为只是模型的结构引起的过拟合，所以修改了dropout的rate，将其变大，也就是丢弃的几率变大。而且还在输出层对参数加了一个L2的正则化。但是几乎没有效果。所以我就意识到，可能不是模型的问题。然后在群上看他们聊天聊到shuffle，我恍然大悟，原来是我读出来的时候没有进行shuffle，现在就好了。


```python
import matplotlib.pyplot as plt

def show_loss(hist, title='loss'):
    # show the training and validation loss
    plt.plot(hist.history['val_loss'], label="validation loss")
    plt.plot(hist.history['loss'], label="train loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(title)
    plt.legend()
    plt.show()
def show_acc(hist, title='accuracy'):
    # show the training and validation loss
    plt.plot(hist.history['val_acc'], label="validation accuracy")
    plt.plot(hist.history['acc'], label="train accuracy")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title(title)
    plt.legend()
    plt.show()

show_loss(hist)
show_acc(hist)
```


![png](output_20_0.png)



![png](output_20_1.png)


模型训练完了之后，应该对测试集进行预测，并且声称csv格式的结果文件提交kaggle，从而得到模型的得分。其实文档的格式很简单，就是对测试集的每张图片的预测结果（也就是这张图片是狗的概率，注意不是0或者1哟！是类似0.99  0.11之类的概率值！），我写了一个专门的接口函数来完成预测以及结果文件的生成


```python
import pandas as pd

def get_test_result(model_obj, test_vec_path, image_size, model_name=""):
    with h5py.File(test_vec_path, 'r') as f:
        x_test = np.array(f['test'])

    pred_test = model_obj.predict(x_test)
    pred_test = pred_test.clip(min=0.005, max=0.995)
    
    df = pd.read_csv("sampleSubmission.csv")

    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(test_link_path, image_size, color_mode='rgb',
                                             shuffle=False, batch_size=64, class_mode=None)

    for i, fname in enumerate(test_generator.filenames):
        index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
        #df.set_value(index-1, 'label', y_pred[i])
        df.loc[index-1, 'label'] = pred_test[i]
    
    df.to_csv('%s.csv' % (model_name), index=None)
    print ('test result file %s.csv generated!' % (model_name))
    df.head(10)

get_test_result(res_top_model, resnet_50_vec_path, (224, 224), model_name="resnet-50")
```

    Found 12500 images belonging to 1 classes.
    test result file resnet-50.csv generated!


这次提交了kaggle之后，loss为0.04274。resnet-50模型先暂时不试了。先把模型参数和参数保存起来。由于是预训练模型，所以就不需要保存resnet-50部分，只保存添加的输出层参数就好了。


```python
res_top_model.save(resnet_50_model_save_name)
```

## 2.2 inceptionv3

接下尝试的是inceptionv3模型。因为以后还会尝试更多的模型，所以这里需要写一个通用的函数，而不是像之前resnet那样很分散。下面这个函数用于各种模型的特征向量的提取。


```python
from keras.layers import Input
from keras.layers.core import Lambda

def model_vector_catch(MODEL, image_size, file_name, preprocessing=None):
    vec_dir = 'vect/'

    input_tensor = Input(shape=(image_size[0], image_size[1], 3))
    if preprocessing:
        ## check if need preprocessing
        input_tensor = Lambda(preprocessing)(input_tensor)
    model_no_top = MODEL(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(image_size[0], image_size[1], 3))
    ## flatten the output shape and generate model
    out = GlobalAveragePooling2D()(model_no_top.output)
    new_model = Model(inputs=model_no_top.input, outputs=out)
    
    ## get iamge generator
    gen = ImageDataGenerator()
    test_gen = ImageDataGenerator()
    """
    classes = ['cat', 'dog'] -- cat is 0, dog is 1, so we need write this
    class_mode = None -- i will not use like 'fit_fitgenerator', so i do not need labels
    shuffle = False -- it is unneccssary
    batch_size = 64 
    """
    train_generator = gen.flow_from_directory(train_link_path, image_size, color_mode='rgb', \
                                              classes=['cat', 'dog'], class_mode=None, shuffle=False, batch_size=64)
    test_generator = test_gen.flow_from_directory(test_link_path, image_size, color_mode='rgb', \
                                          class_mode=None, shuffle=False, batch_size=64)
    """
    steps = None, by default, the steps = len(generator)
    """
    train_vector = new_model.predict_generator(train_generator)
    test_vector = new_model.predict_generator(test_generator)
    
    with h5py.File(vec_dir + ("%s.h5" % (file_name)), 'w') as f: 
        f.create_dataset('x_train', data=train_vector)
        f.create_dataset("y_train", data=train_generator.classes)
        f.create_dataset("test", data=test_vector)
    print ("Model %s vector cached complete!" % (file_name))
```


```python
from keras.applications import inception_v3

inceptionv3_vec_path = 'vect/inceptionv3.h5'

if not isfile(inceptionv3_vec_path):
    model_vector_catch(inception_v3.InceptionV3, (299, 299), 'inceptionv3', inception_v3.preprocess_input)
```

    Found 24975 images belonging to 2 classes.
    Found 12500 images belonging to 1 classes.
    Model inceptionv3 vector cached complete!


同样，我们先搭建模型，然后进行迁移学习。由于inceptionv3在全连接层之前的输出维度是(?, 1, 1, 2048)，所以新搭建的模型的输入维度是2048.

inception v3论文上说其训练时候采用的是RMSProp，初始学习速率为0.045，rho=0.9, epsilon=1.0（确定吗？但是论文上是这样写的），decay采用指数衰减，每两个epoch衰减一次。指数衰减的公式如下：decayed_learning_rate = lr_base * decay_rate ^ (global_step / decay_steps)  

但是我有一个问题，论文中给出的训练的建议是基于ImageNet的训练集，并且作者是训练整体的模型，但是现在是在猫狗数据上进行训练，而且并没有对整体的模型进行训练，那我们是否应该采用论文中的建议的训练方式，下面尝试一下就知道了。


```python
import numpy as np
from sklearn.utils import shuffle

with h5py.File(inceptionv3_vec_path, 'r') as f:
    x_train = np.array(f['x_train'])
    y_train = np.array(f['y_train'])
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
```


```python
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler
import math
    
input_tensor = Input(shape=(2048,))
x = Dropout(0.5)(input_tensor)
x = Dense(1, activation='sigmoid', name='inc_dense_1')(x)

inceptionv3_model = Model(inputs=input_tensor, outputs=x)

#optimizer
#lr_base = 0.045
lr_base = 0.0002
decay_rate = 0.94
decay_steps = 2
#opt = RMSprop(lr=lr_base, rho=0.9, epsilon=1.0)
opt = RMSprop(lr=lr_base, rho=0.9)

# exponential rate decay:decayed_learning_rate = lr_base * decay_rate ^ (global_step / decay_steps)
def lr_scheduler(epoch):
    # calculate the new learning rate according to epoch number
    lr = lr_base * ((decay_rate)**math.floor(epoch/decay_steps))
    print ("Epoch %d , new lr==%f" % (epoch, lr))    
    
    return lr

scheduler = LearningRateScheduler(lr_scheduler)
inceptionv3_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

hist = inceptionv3_model.fit(x_train, y_train, batch_size=32, epochs=30, validation_split=0.2, callbacks=[scheduler])
```

    Train on 19980 samples, validate on 4995 samples
    Epoch 1/30
    Epoch 0 , new lr==0.000200
    19980/19980 [==============================] - 4s 221us/step - loss: 0.1552 - acc: 0.9556 - val_loss: 0.0452 - val_acc: 0.9888
    Epoch 2/30
    Epoch 1 , new lr==0.000200
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0408 - acc: 0.9901 - val_loss: 0.0314 - val_acc: 0.9902
    Epoch 3/30
    Epoch 2 , new lr==0.000188
    19980/19980 [==============================] - 2s 108us/step - loss: 0.0322 - acc: 0.9908 - val_loss: 0.0269 - val_acc: 0.9914
    Epoch 4/30
    Epoch 3 , new lr==0.000188
    19980/19980 [==============================] - 2s 110us/step - loss: 0.0279 - acc: 0.9919 - val_loss: 0.0250 - val_acc: 0.9916
    Epoch 5/30
    Epoch 4 , new lr==0.000177
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0258 - acc: 0.9930 - val_loss: 0.0241 - val_acc: 0.9916
    Epoch 6/30
    Epoch 5 , new lr==0.000177
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0239 - acc: 0.9930 - val_loss: 0.0231 - val_acc: 0.9920
    Epoch 7/30
    Epoch 6 , new lr==0.000166
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0236 - acc: 0.9932 - val_loss: 0.0230 - val_acc: 0.9922
    Epoch 8/30
    Epoch 7 , new lr==0.000166
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0222 - acc: 0.9935 - val_loss: 0.0231 - val_acc: 0.9918
    Epoch 9/30
    Epoch 8 , new lr==0.000156
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0216 - acc: 0.9930 - val_loss: 0.0228 - val_acc: 0.9920
    Epoch 10/30
    Epoch 9 , new lr==0.000156
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0225 - acc: 0.9931 - val_loss: 0.0239 - val_acc: 0.9918
    Epoch 11/30
    Epoch 10 , new lr==0.000147
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0225 - acc: 0.9932 - val_loss: 0.0218 - val_acc: 0.9926
    Epoch 12/30
    Epoch 11 , new lr==0.000147
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0233 - acc: 0.9922 - val_loss: 0.0220 - val_acc: 0.9928
    Epoch 13/30
    Epoch 12 , new lr==0.000138
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0208 - acc: 0.9933 - val_loss: 0.0223 - val_acc: 0.9922
    Epoch 14/30
    Epoch 13 , new lr==0.000138
    19980/19980 [==============================] - 2s 108us/step - loss: 0.0212 - acc: 0.9934 - val_loss: 0.0215 - val_acc: 0.9934
    Epoch 15/30
    Epoch 14 , new lr==0.000130
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0189 - acc: 0.9944 - val_loss: 0.0215 - val_acc: 0.9930
    Epoch 16/30
    Epoch 15 , new lr==0.000130
    19980/19980 [==============================] - 2s 108us/step - loss: 0.0207 - acc: 0.9935 - val_loss: 0.0215 - val_acc: 0.9930
    Epoch 17/30
    Epoch 16 , new lr==0.000122
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0207 - acc: 0.9931 - val_loss: 0.0213 - val_acc: 0.9932
    Epoch 18/30
    Epoch 17 , new lr==0.000122
    19980/19980 [==============================] - 2s 108us/step - loss: 0.0202 - acc: 0.9938 - val_loss: 0.0221 - val_acc: 0.9926
    Epoch 19/30
    Epoch 18 , new lr==0.000115
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0207 - acc: 0.9933 - val_loss: 0.0221 - val_acc: 0.9926
    Epoch 20/30
    Epoch 19 , new lr==0.000115
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0199 - acc: 0.9941 - val_loss: 0.0213 - val_acc: 0.9932
    Epoch 21/30
    Epoch 20 , new lr==0.000108
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0197 - acc: 0.9941 - val_loss: 0.0214 - val_acc: 0.9932
    Epoch 22/30
    Epoch 21 , new lr==0.000108
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0199 - acc: 0.9939 - val_loss: 0.0216 - val_acc: 0.9932
    Epoch 23/30
    Epoch 22 , new lr==0.000101
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0194 - acc: 0.9937 - val_loss: 0.0214 - val_acc: 0.9936
    Epoch 24/30
    Epoch 23 , new lr==0.000101
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0178 - acc: 0.9944 - val_loss: 0.0215 - val_acc: 0.9932
    Epoch 25/30
    Epoch 24 , new lr==0.000095
    19980/19980 [==============================] - 2s 108us/step - loss: 0.0199 - acc: 0.9939 - val_loss: 0.0215 - val_acc: 0.9932
    Epoch 26/30
    Epoch 25 , new lr==0.000095
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0203 - acc: 0.9935 - val_loss: 0.0215 - val_acc: 0.9934
    Epoch 27/30
    Epoch 26 , new lr==0.000089
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0192 - acc: 0.9943 - val_loss: 0.0218 - val_acc: 0.9932
    Epoch 28/30
    Epoch 27 , new lr==0.000089
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0204 - acc: 0.9938 - val_loss: 0.0220 - val_acc: 0.9932
    Epoch 29/30
    Epoch 28 , new lr==0.000084
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0211 - acc: 0.9938 - val_loss: 0.0216 - val_acc: 0.9936
    Epoch 30/30
    Epoch 29 , new lr==0.000084
    19980/19980 [==============================] - 2s 109us/step - loss: 0.0197 - acc: 0.9945 - val_loss: 0.0216 - val_acc: 0.9934



```python
show_loss(hist)
show_acc(hist)
```


![png](output_31_0.png)



![png](output_31_1.png)


可以看到，如果按照论文上的训练方法训练效果还是很不错的，现在就来看看该模型的评分吧！


```python
get_test_result(inceptionv3_model, inceptionv3_vec_path, (299, 299), model_name="inceptionv3")
```

    Found 12500 images belonging to 1 classes.
    test result file inceptionv3.csv generated!


提交kaggle之后，发现获得的分数为0.04074，在leaderboard中可以排名到第17名，也就是在2%以内。同样，我们保存该模型与权重。


```python
inceptionv3_model.save(inceptionv3_model_save_name)
```

## 2.3 xception
接下来我将尝试xception模型，出了预处理函数和图片上输出尺寸需要改动外，其他均和上述过程一致


```python
from keras.applications import xception

xception_vec_path = 'vect/xception.h5'

if not isfile(xception_vec_path):
    model_vector_catch(xception.Xception, (299, 299), 'xception', xception.preprocess_input)
```

    Found 24975 images belonging to 2 classes.
    Found 12500 images belonging to 1 classes.
    Model xception vector cached complete!



```python
with h5py.File(xception_vec_path, 'r') as f:
    x_train = np.array(f['x_train'])
    y_train = np.array(f['y_train'])
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
```

对于训练方式，我还是选择与inceptionv3一样的训练方式


```python
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler
import math
    
input_tensor = Input(shape=(2048,))
x = Dropout(0.5)(input_tensor)
x = Dense(1, activation='sigmoid', name='xce_dense_1')(x)

xception_model = Model(inputs=input_tensor, outputs=x)

#optimizer
#lr_base = 0.045
lr_base = 0.0002
decay_rate = 0.94
decay_steps = 2
#opt = RMSprop(lr=lr_base, rho=0.9, epsilon=1.0)
opt = RMSprop(lr=lr_base, rho=0.9)

# exponential rate decay:decayed_learning_rate = lr_base * decay_rate ^ (global_step / decay_steps)
def lr_scheduler(epoch):
    # calculate the new learning rate according to epoch number
    lr = lr_base * ((decay_rate)**math.floor(epoch/decay_steps))
    print ("Epoch %d , new lr==%f" % (epoch, lr))    
    
    return lr

scheduler = LearningRateScheduler(lr_scheduler)
xception_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

hist = xception_model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.2, callbacks=[scheduler])
```

    Train on 19980 samples, validate on 4995 samples
    Epoch 1/20
    Epoch 0 , new lr==0.000200
    19980/19980 [==============================] - 5s 261us/step - loss: 0.1810 - acc: 0.9600 - val_loss: 0.0474 - val_acc: 0.9912
    Epoch 2/20
    Epoch 1 , new lr==0.000200
    19980/19980 [==============================] - 2s 112us/step - loss: 0.0385 - acc: 0.9915 - val_loss: 0.0311 - val_acc: 0.9920
    Epoch 3/20
    Epoch 2 , new lr==0.000188
    19980/19980 [==============================] - 2s 112us/step - loss: 0.0284 - acc: 0.9926 - val_loss: 0.0271 - val_acc: 0.9920
    Epoch 4/20
    Epoch 3 , new lr==0.000188
    19980/19980 [==============================] - 2s 112us/step - loss: 0.0259 - acc: 0.9927 - val_loss: 0.0258 - val_acc: 0.9924
    Epoch 5/20
    Epoch 4 , new lr==0.000177
    19980/19980 [==============================] - 2s 112us/step - loss: 0.0232 - acc: 0.9931 - val_loss: 0.0246 - val_acc: 0.9924
    Epoch 6/20
    Epoch 5 , new lr==0.000177
    19980/19980 [==============================] - 2s 112us/step - loss: 0.0221 - acc: 0.9932 - val_loss: 0.0242 - val_acc: 0.9926
    Epoch 7/20
    Epoch 6 , new lr==0.000166
    19980/19980 [==============================] - 2s 111us/step - loss: 0.0219 - acc: 0.9938 - val_loss: 0.0240 - val_acc: 0.9926
    Epoch 8/20
    Epoch 7 , new lr==0.000166
    19980/19980 [==============================] - 2s 113us/step - loss: 0.0210 - acc: 0.9940 - val_loss: 0.0236 - val_acc: 0.9928
    Epoch 9/20
    Epoch 8 , new lr==0.000156
    19980/19980 [==============================] - 2s 112us/step - loss: 0.0206 - acc: 0.9938 - val_loss: 0.0234 - val_acc: 0.9930
    Epoch 10/20
    Epoch 9 , new lr==0.000156
    19980/19980 [==============================] - 2s 112us/step - loss: 0.0204 - acc: 0.9940 - val_loss: 0.0233 - val_acc: 0.9930
    Epoch 11/20
    Epoch 10 , new lr==0.000147
    19980/19980 [==============================] - 2s 112us/step - loss: 0.0191 - acc: 0.9943 - val_loss: 0.0232 - val_acc: 0.9932
    Epoch 12/20
    Epoch 11 , new lr==0.000147
    19980/19980 [==============================] - 2s 112us/step - loss: 0.0201 - acc: 0.9942 - val_loss: 0.0234 - val_acc: 0.9932
    Epoch 13/20
    Epoch 12 , new lr==0.000138
    19980/19980 [==============================] - 2s 112us/step - loss: 0.0199 - acc: 0.9939 - val_loss: 0.0231 - val_acc: 0.9932
    Epoch 14/20
    Epoch 13 , new lr==0.000138
    19980/19980 [==============================] - 2s 112us/step - loss: 0.0185 - acc: 0.9941 - val_loss: 0.0232 - val_acc: 0.9932
    Epoch 15/20
    Epoch 14 , new lr==0.000130
    19980/19980 [==============================] - 2s 111us/step - loss: 0.0180 - acc: 0.9943 - val_loss: 0.0233 - val_acc: 0.9932
    Epoch 16/20
    Epoch 15 , new lr==0.000130
    19980/19980 [==============================] - 2s 111us/step - loss: 0.0192 - acc: 0.9941 - val_loss: 0.0231 - val_acc: 0.9934
    Epoch 17/20
    Epoch 16 , new lr==0.000122
    19980/19980 [==============================] - 2s 111us/step - loss: 0.0189 - acc: 0.9941 - val_loss: 0.0230 - val_acc: 0.9934
    Epoch 18/20
    Epoch 17 , new lr==0.000122
    19980/19980 [==============================] - 2s 111us/step - loss: 0.0188 - acc: 0.9944 - val_loss: 0.0230 - val_acc: 0.9934
    Epoch 19/20
    Epoch 18 , new lr==0.000115
    19980/19980 [==============================] - 2s 111us/step - loss: 0.0186 - acc: 0.9937 - val_loss: 0.0232 - val_acc: 0.9934
    Epoch 20/20
    Epoch 19 , new lr==0.000115
    19980/19980 [==============================] - 2s 111us/step - loss: 0.0191 - acc: 0.9940 - val_loss: 0.0231 - val_acc: 0.9934


可视化fit过程


```python
show_loss(hist)
show_acc(hist)
```


![png](output_42_0.png)



![png](output_42_1.png)



```python
get_test_result(xception_model, xception_vec_path, (299, 299), model_name="xception")
```

    Found 12500 images belonging to 1 classes.
    test result file xception.csv generated!


xception在测试集上的分数是0.04138，结果和inceptionv3差不多


```python
xception_model.save(xception_model_save_name)
```


```python
tmp = xception_model.get_layer(name='xce_dense_1')
print (tmp.get_weights())
```

    [array([[-0.05352252],
           [ 0.03235795],
           [-0.03075365],
           ...,
           [ 0.1362454 ],
           [-0.05877006],
           [-0.14424288]], dtype=float32), array([0.08456017], dtype=float32)]


# 3 模型的集成

我们知道，在机器学习中有一种技巧叫做集成学习，不知道其是否适用于深度学习，集成学习的思想就是三个臭皮匠顶个诸葛亮，让多个模型来进行投票，遵循少数服从多数的原则来产生预测结果，可以使预测结果更加平滑，准确，坏处就是可能会增加预测的时间，但是既然kaggle没有提到有模型预测时间的要求，那么我们就可以尝试一些集成学习。

在这里我简单地尝试一下uniform blending的集成方式，然后采用取三个模型的输出值的平均值作为集成的最终输出（而不是所谓的‘投票’，因为kaggle需要输出概率值而不是投票结果）。

注意：uniform blending集成方法的假设是我们的三个基准模型本身拥有diversity，这样才会有好的结果。我不清楚我的这三个模型知否符合这样的要求。而且集成方法的效果总是会随着基准模型的数量的增多而变好，所以我也不清楚3个模型到底够不够。既然不懂，就尝试嘛！

创建集成模型，由于网络结构比较简答，就没有必要像之前的bottleneck features那样预先保存向量了，直接将之前训练好的三个模型的输出层合并求平均就好了。

定义一个函数，方便获取各个不同基准模型的bottleneck features。
现在再获取各个基准模型的bottleneck features，作为集成模型的输入


```python
def get_bottleneck_features(path):
    ## get bottleneck features
    with h5py.File(path, 'r') as f:
        x_train = np.array(f['x_train'])
        y_train = np.array(f['y_train'])
        x_test = np.array(f['test'])
        return x_train,  y_train, x_test

## resnet 50
res_x_train, res_y_train, res_test = get_bottleneck_features(resnet_50_vec_path)

## inception v3
inc_x_train, inc_y_train, inc_test = get_bottleneck_features(inceptionv3_vec_path)

## xception
xce_x_train, xce_y_train, xce_test = get_bottleneck_features(xception_vec_path)
```


```python
from keras.layers import Average
from keras.models import load_model

## load saved model(include weights)
model_res = load_model(resnet_50_model_save_name)
model_inc = load_model(inceptionv3_model_save_name)
model_xce = load_model(xception_model_save_name)

## resnet_50
resnet50_input = model_res.input
resnet50_output = model_res.output

##inception_v3
inceptionv3_input = model_inc.input
inceptionv3_output = model_inc.output

##xception
xception_input = model_xce.input
xception_output = model_xce.output

agg_out = Average()([resnet50_output, inceptionv3_output, xception_output])

agg_model = Model(inputs=[resnet50_input, inceptionv3_input, xception_input], outputs=agg_out)
```

因为没有任何参数需要优化，所以无需训练，我们现在evaluation来看看结果如何


```python
agg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```


```python
agg_model.evaluate([res_x_train, inc_x_train, xce_x_train], res_y_train, batch_size=32)
```

    24975/24975 [==============================] - 4s 142us/step





    [0.013857072906072687, 0.9961961961961961]




```python
def agg_get_test_result(model_obj, input_list, model_name=""):
    ## predict test sample
    pred_test = model_obj.predict(input_list)
    pred_test = pred_test.clip(min=0.005, max=0.995)
    
    ## save the reulst to file
    df = pd.read_csv("sampleSubmission.csv")

    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(test_link_path, color_mode='rgb',
                                             shuffle=False, batch_size=64, class_mode=None)

    for i, fname in enumerate(test_generator.filenames):
        index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
        #df.set_value(index-1, 'label', y_pred[i])
        df.loc[index-1, 'label'] = pred_test[i]
    
    df.to_csv('%s.csv' % (model_name), index=None)
    print ('test result file %s.csv generated!' % (model_name))
    df.head(10)
```


```python
agg_get_test_result(agg_model, [res_test, inc_test, xce_test], model_name="aggregation")
```

    Found 12500 images belonging to 1 classes.
    test result file aggregation.csv generated!


提交kaggle之后，loss为0.03949，突破了0.04了，果然是要比单模型效果好。


```python
agg_get_test_result(agg_model, [res_test, inc_test, xce_test], model_name="aggregation")
```

    Found 12500 images belonging to 1 classes.
    test result file aggregation.csv generated!


所以最终我选择集成的模型的方式。

# 4. 最终的模型


```python
from keras.models import load_model

res_tmp_model = load_model(resnet_50_model_save_name)
inc_tmp_model = load_model(inceptionv3_model_save_name)
xce_tmp_model = load_model(xception_model_save_name)
```


```python
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.applications import xception
from keras.layers import Average


##build aggregation model
resNet_input_shape = (224,224,3)
inc_input_shape = (299, 299, 3)
xce_input_shape = (299, 299, 3)

"""
resnet_50
"""
#load raw resnet50 model(without top layer)
agg_res_input = Input(shape=resNet_input_shape)
agg_res_x = Lambda(resnet50.preprocess_input)(agg_res_input)
agg_res_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=agg_res_x, input_shape=resNet_input_shape)
agg_res_x = GlobalAveragePooling2D()(agg_res_model.output)
# load pretrained model and weights
res_tmp_layer = res_tmp_model.get_layer(name='res_dense_1')
res_tmp_weights = res_tmp_layer.get_weights()
# set weights
agg_res_tmp_layer = Dense(1, activation='sigmoid', name='agg_res_dense_1')
agg_res_x = agg_res_tmp_layer(agg_res_x)
agg_res_tmp_layer.set_weights(res_tmp_weights)


"""
inception v3
"""
agg_inc_input = Input(shape=inc_input_shape)
agg_inc_x = Lambda(inception_v3.preprocess_input)(agg_inc_input)
agg_inc_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=agg_inc_x, input_shape=inc_input_shape)
agg_inc_x = GlobalAveragePooling2D()(agg_inc_model.output)
# load pretrained model and weights
inc_tmp_layer = inc_tmp_model.get_layer(name='inc_dense_1')
inc_tmp_weights = inc_tmp_layer.get_weights()
# set weights
agg_inc_tmp_layer = Dense(1, activation='sigmoid', name='agg_inc_dense_1')
agg_inc_x = agg_inc_tmp_layer(agg_inc_x)
agg_inc_tmp_layer.set_weights(inc_tmp_weights)

"""
xception
"""
agg_xce_input = Input(shape=xce_input_shape)
agg_xce_x = Lambda(xception.preprocess_input)(agg_xce_input)
agg_xce_model = xception.Xception(include_top=False, weights='imagenet', input_tensor=agg_xce_x, input_shape=xce_input_shape)
agg_xce_x = GlobalAveragePooling2D()(agg_xce_model.output)
# load pretrained model and weights
xce_tmp_layer = xce_tmp_model.get_layer(name='xce_dense_1')
xce_tmp_weights = xce_tmp_layer.get_weights()
# set weights
agg_xce_tmp_layer = Dense(1, activation='sigmoid', name='agg_xce_dense_1')
agg_xce_x = agg_xce_tmp_layer(agg_xce_x)
agg_xce_tmp_layer.set_weights(xce_tmp_weights)

"""
merge three model's output
"""
agg_out = Average()([agg_res_x, agg_inc_x, agg_xce_x])

"""
create model
"""
model = Model(inputs=[agg_res_input, agg_inc_input, agg_xce_input], outputs=agg_out)
```


```python
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cv2
import numpy as np
import sys
import time
%matplotlib inline

def predict(picture_path_list, dis_msg=True):
    results = []
    
    for picture in picture_path_list:
        start = time.time()
        ## read image and change channel order
        image = cv2.imread(picture)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        ## resize image
        res_img_input = cv2.resize(image, (resNet_input_shape[0], resNet_input_shape[1]))
        inc_img_input = cv2.resize(image, (inc_input_shape[0], inc_input_shape[1]))
        xce_img_input = cv2.resize(image, (xce_input_shape[0], xce_input_shape[1]))

        ## expand dimensions
        res_img_input= np.expand_dims(res_img_input,axis=0)
        inc_img_input= np.expand_dims(inc_img_input,axis=0)
        xce_img_input= np.expand_dims(xce_img_input,axis=0)

        ## predict result
        result = model.predict([res_img_input, inc_img_input, xce_img_input])

        end = time.time()
        
        if dis_msg == True:
            plt.imshow(image)

            if result >= 0.5:
                plt.title('Time cost %f seconds\nIt is a dog, Probability:%f' % (end-start, result))
            else:
                plt.title('Time cost %f seconds\nIt is a cat, Probability:%f' % (end-start, 1-result))
            ##可以取消坐标的显示
            plt.axis("off")
        
        results.append(result[0])

    return results
```


```python
predict(['train/cat.0.jpg'])
```

# 5.误差分析


```python
from tqdm import tqdm  
import numpy as np

image_num = 12500

def predict_arr(res_arr, inc_arr, xce_arr):
    predict = model.predict([res_arr, inc_arr, xce_arr])
    predict = predict.reshape(len(predict))
    return list(predict)

def predict_dir(dir_path, batch_size, is_dog=True):
    count = 0
    index = 0
    predict_point = np.arange(0, image_num, batch_size)
    predict_point = predict_point[1:]
    
    ## get name prefix
    if is_dog == True:
        name = 'dog.'
    else:
        name = 'cat.'
        
    ## batch image array
    res_arr = np.zeros((batch_size,) + resNet_input_shape, dtype=np.uint8)
    inc_arr = np.zeros((batch_size,) + inc_input_shape, dtype=np.uint8)
    xce_arr = np.zeros((batch_size,) + xce_input_shape, dtype=np.uint8)
    ## all image predict results
    results = []
    
    for i in tqdm(range(image_num)):
        if i in predict_point:
            ## predict batch image
            results = results + predict_arr(res_arr, inc_arr, xce_arr)
            
        ## get image from file
        image = cv2.imread(dir_path + '/' + name + str(i) + '.jpg')
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        index = i%batch_size
        res_arr[index, ...] = cv2.resize(image, (resNet_input_shape[0], resNet_input_shape[1]))
        inc_arr[index, ...] = cv2.resize(image, (inc_input_shape[0], inc_input_shape[1]))
        xce_arr[index, ...] = cv2.resize(image, (xce_input_shape[0], xce_input_shape[1]))
    
    ## predict the rest
    result = predict_arr(res_arr[:index+1], inc_arr[:index+1], xce_arr[:index+1])
    results = results + predict_arr(res_arr[:index+1], inc_arr[:index+1], xce_arr[:index+1])

    return results
```


```python
cat_results = predict_dir(train_link_cat, 64, False)
dog_results = predict_dir(train_link_dog, 64)
print (len(dog_results))
```

    100%|██████████| 5000/5000 [04:04<00:00, 20.41it/s]
    100%|██████████| 5000/5000 [04:06<00:00, 20.28it/s]


    5000



```python
cat_results = np.array(cat_results)
cat_results_mask = cat_results >= 0.5

dog_results = np.array(dog_results)
dog_results_mask = dog_results < 0.5

print ('cat Error num: ' + str(sum(cat_results_mask)))
print ('dog Error num: ' + str(sum(dog_results_mask)))

cat_error_index = np.arange(image_num)[cat_results_mask]
dog_error_index = np.arange(image_num)[dog_results_mask]
```

    cat Error num: 14
    dog Error num: 18



```python
import matplotlib.pyplot as plt
import cv2
import numpy as np
%matplotlib inline

cat_error_name = ['cat.'+str(i)+'.jpg' for i in cat_error_index]
dog_error_name = ['dog.'+str(i)+'.jpg' for i in dog_error_index]

plt.figure(1, figsize=(13, 13))

for i,filename in enumerate(dog_error_name[:]):
    image = cv2.imread('train/'+str(filename))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    ax1=plt.subplot(5,5,i+1)
    plt.imshow(image)
    plt.axis("off")
    plt.title(filename)
    
plt.show()
```


![png](output_69_0.png)



```python
plt.figure(1, figsize=(13, 13))

for i,filename in enumerate(cat_error_name[:]):
    image = cv2.imread('train/'+str(filename))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    ax1=plt.subplot(5,5,i+1)
    plt.imshow(image)
    plt.axis("off")
    plt.title(filename)
    
plt.show()
```


![png](output_70_0.png)


上面的结果是我在12500张猫和狗的训练集图片中找出的预测错误的图片，可以看出这些错误分类的图片可以分成几类：


1.照片的主体不明确，有些照片中猫或者狗并不是主体，比如在cat.724和cat.3672中，显然主体是狗和人，这让算法很难学习。而且像cat.4085本身就是一只狗。

2.人眼无法辨认是猫或者狗。比如cat.3105其实看起来像是一条狗。比如dog.2877看起来确像是一只猫。

所以我们接下来的方向可以是做一些数据异常值清理。比如cat.4085可以现在直接清理掉，因为这是一个错误的标记。还有dog.2877也是需要清理的，因为这看起来是一只猫。

同时，接下来要做的事情参考了[凌蓝风](https://zhuanlan.zhihu.com/p/34068451)的知乎文章，我感觉他的思路很好。

我们的任务是猫狗分类，然后在ImageNet上进行预训练的模型也支持识别猫狗，所以我们可以根据ImageNet上的预训练模型的TOP-N的结果来分辨一张图片是不是异常的样本。

什么是TOP-N呢，我们知道ImageNet上的预训练模型的输出有1000个类别，TOP-N就是在模型的输出1000类别的概率中选择概率最高的N个类别。

比如将猫狗训练集的一张图片输入预训练模型，如果在TOP-1或者TOP-5中没有猫或者狗，这情有可原。但是如果在TOP-10或者TOP-20的结果中都没有猫狗的话就说不过去了，说明这张训练集的图片猫狗的主题根本不是猫狗，或者压根就没有猫狗，我们根据这样的结果来选择异常的样本。

# 6.1 使用InceptionResNetV2进行异常值清理


```python
from keras.applications import inception_resnet_v2

## resNet-50 do not need preprocessing, so the resNet_input_shape is not neccessary
clean_model_input_shape = (299,299)

inc_res_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet')
```


```python
dogs_map = [
 'n02085620','n02085782','n02085936','n02086079'
,'n02086240','n02086646','n02086910','n02087046'
,'n02087394','n02088094','n02088238','n02088364'
,'n02088466','n02088632','n02089078','n02089867'
,'n02089973','n02090379','n02090622','n02090721'
,'n02091032','n02091134','n02091244','n02091467'
,'n02091635','n02091831','n02092002','n02092339'
,'n02093256','n02093428','n02093647','n02093754'
,'n02093859','n02093991','n02094114','n02094258'
,'n02094433','n02095314','n02095570','n02095889'
,'n02096051','n02096177','n02096294','n02096437'
,'n02096585','n02097047','n02097130','n02097209'
,'n02097298','n02097474','n02097658','n02098105'
,'n02098286','n02098413','n02099267','n02099429'
,'n02099601','n02099712','n02099849','n02100236'
,'n02100583','n02100735','n02100877','n02101006'
,'n02101388','n02101556','n02102040','n02102177'
,'n02102318','n02102480','n02102973','n02104029'
,'n02104365','n02105056','n02105162','n02105251'
,'n02105412','n02105505','n02105641','n02105855'
,'n02106030','n02106166','n02106382','n02106550'
,'n02106662','n02107142','n02107312','n02107574'
,'n02107683','n02107908','n02108000','n02108089'
,'n02108422','n02108551','n02108915','n02109047'
,'n02109525','n02109961','n02110063','n02110185'
,'n02110341','n02110627','n02110806','n02110958'
,'n02111129','n02111277','n02111500','n02111889'
,'n02112018','n02112137','n02112350','n02112706'
,'n02113023','n02113186','n02113624','n02113712'
,'n02113799','n02113978']

cats_map=[
'n02123045','n02123159','n02123394','n02123597'
,'n02124075','n02125311','n02127052']

image_num = 12500

def data_clean_predict_dir(pre_trained_model, dir_path, shape, preprocess, decode, is_dog=True):
    results = []
    
    if is_dog == True:
        name = 'dog.'
        check = dogs_map
    else:
        name = 'cat.'
        check = cats_map
    
    for i in tqdm(range(image_num)):
        ## read image and change channel order
        #print (dir_path + '/'+name +  + str(i) + '.jpg')
        image = cv2.imread(dir_path + '/'+ name + str(i) + '.jpg')
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        ## resize image
        input_data = cv2.resize(image, shape)

        ## expand dimensions
        input_data= np.expand_dims(input_data,axis=0)

        ## predict result
        input_data = input_data.astype(np.float64)
        input_data = preprocess(input_data)
        
        preds = pre_trained_model.predict(input_data)
        
        #print('Predicted:', decode(preds, top=3)[0])
        tmp = decode(preds, top=60)[0]
        
        ##check if the picure is in the list
        tmp = [item[0] for item in tmp]
        if len(list(set(tmp).intersection(set(check)))) == 0:
            results.append(name + str(i) + '.jpg')

    return results

def data_clean_display(name_list):
    plt.figure(1, figsize=(18, 18))
    
    ## 
    if (len(name_list) > 40):
        name_list = name_list[0:25]
    
    for i,filename in enumerate(name_list):
        image = cv2.imread('train/'+str(filename))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        ax1=plt.subplot(5,8,i+1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(filename)
    
    plt.show()
```

## 清理猫图片的异常样本


```python
dog_name_list = data_clean_predict_dir(inc_res_model, train_link_cat, (150,150), inception_resnet_v2.preprocess_input,\
                       inception_resnet_v2.decode_predictions, is_dog=False)
```

    100%|██████████| 12500/12500 [07:03<00:00, 29.54it/s]



```python
cat_name_list = dog_name_list
data_clean_display(cat_name_list)
```


![png](output_79_0.png)


上述结果中，每一张都有猫，只不过是背景复杂了点儿，或者猫的主体小了点儿，但是都不应该被清理。我感觉该模型的筛选结果不是很理想，所以我不打算使用它的结果

## 清理狗图片的异常样本


```python
dog_name_list = data_clean_predict_dir(inc_res_model, train_link_dog, (150,150), inception_resnet_v2.preprocess_input,\
                       inception_resnet_v2.decode_predictions, is_dog=True)
```

    100%|██████████| 12500/12500 [06:58<00:00, 29.89it/s]



```python
data_clean_display(dog_name_list)
```


![png](output_83_0.png)


# 6.2 使用resnet50进行异常值清理


```python
from keras.applications import resnet50

model = resnet50.ResNet50(weights='imagenet')
```


```python
res_cat_name_list = data_clean_predict_dir(model, train_link_cat, (224,224), resnet50.preprocess_input,\
                       resnet50.decode_predictions, is_dog=False)
```

    100%|██████████| 12500/12500 [03:00<00:00, 69.09it/s]



```python
data_clean_display(res_cat_name_list)
```


![png](output_87_0.png)


对于resnet的狗的筛选结果，我选出来了几张我认为不利于算法学习的图片，比如3672,3216,2520


```python
res_dog_name_list = data_clean_predict_dir(model, train_link_dog, (224,224), resnet50.preprocess_input,\
                       resnet50.decode_predictions, is_dog=True)
```

    100%|██████████| 12500/12500 [02:55<00:00, 71.10it/s]



```python
data_clean_display(res_dog_name_list)
```


![png](output_90_0.png)


resnet-50关于dog的筛选结果非常理想，我感觉每一张图片都应该被删除掉。

## 6.3 使用xception进行异常值清理


```python
from keras.applications import xception

model = xception.Xception(weights='imagenet')
```

    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5
    91889664/91884032 [==============================] - 1s 0us/step



```python
xce_cat_name_list = data_clean_predict_dir(model, train_link_cat, (299,299), xception.preprocess_input,\
                       xception.decode_predictions, is_dog=False)
```

    100%|██████████| 12500/12500 [02:55<00:00, 71.09it/s]



```python
data_clean_display(xce_cat_name_list)
```


![png](output_95_0.png)



```python
xce_dog_name_list = data_clean_predict_dir(model, train_link_dog, (299,299), xception.preprocess_input,\
                       xception.decode_predictions, is_dog=True)
```

    100%|██████████| 12500/12500 [02:52<00:00, 72.29it/s]



```python
data_clean_display(xce_dog_name_list)
```


![png](output_97_0.png)


上述结果明显都应该被清除掉


```python


clean_cats_list = ['cat.5351.jpg','cat.4338','cat.4688','cat.5418','cat.9171','cat.8456','cat.7968','cat.7564','cat.7377','cat.10029','cat.10712','cat.11184','cat.12272']
clean_dogs_list = list(set(res_dog_name_list + xce_dog_name_list))

print ("Clean %d cats, and %d dogs" % (len(clean_cats_list), len(clean_dogs_list)))
```

    Clean 13 cats, and 24 dogs


接下来要把这些名字存入硬盘，因为我们要返回第一步开始重新训练模型。


```python
np.save("clean_cats_list.npy",clean_cats_list)
np.save("clean_dogs_list.npy",clean_dogs_list)
```

在做完了异常检测时候，最终的集成模型的得分提升了0.0004zu
