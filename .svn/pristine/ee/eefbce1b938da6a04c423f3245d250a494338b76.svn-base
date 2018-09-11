import scipy.io as sio
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# 该程序为将印刷体数字数据集zhf.mat放入到下面的网络模型中进行训练，得到my_model.h5模型，供数字识别使用


# python读取.mat文件以及处理得到的结果
load_fn = 'zhf.mat'
data_train = sio.loadmat(load_fn)  # 读取的数据集合  10000张图 7000张训练 3000张测试

# batch_size 太小会导致训练慢，过拟合等问题，太大会导致欠拟合。所以要适当选择
batch_size = 128
# 0-9手写数字一个有10个类别
num_classes = 10
# 12次完整迭代
epochs = 12
# input image dimensions输入的图片是28*28像素的灰度图
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets训练集，测试集收集
x_train = data_train['train_x']   # 训练的图片集
y_train = data_train['train_y']   # 训练的标签集
x_test = data_train['test_x']     # 测试的图片集
y_test = data_train['test_y']     # 测试的标签集
# keras输入数据有两种格式，一种是通道数放在前面，一种是通道数放在后面，只是格式差别
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# 把数据变成float32更精确
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices把类别0-9变成2进制，方便训练
# y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
# Sequential类可以让我们灵活地插入不同的神经网络层
model = Sequential()
# 加上一个2D卷积层， 32个输出（也就是卷积通道），激活函数选用relu，
# 卷积核的窗口选用3*3像素窗口
model.add(Conv2D(32,
                 activation='relu',
                 input_shape=input_shape,
                 nb_row=3,
                 nb_col=3))
# 64个通道的卷积层
model.add(Conv2D(64, activation='relu',
                 nb_row=3,
                 nb_col=3))
# 池化层是2*2像素的
model.add(MaxPooling2D(pool_size=(2, 2)))
# 对于池化层的输出，采用0.35概率的Dropout
model.add(Dropout(0.35))
# 展平所有像素，比如[28*28] -> [784]
model.add(Flatten())
# 对所有像素使用全连接层，输出为128，激活函数选用relu
model.add(Dense(128, activation='relu'))
# 对输入采用0.5概率的Dropout
model.add(Dropout(0.5))
# 对刚才Dropout的输出采用softmax激活函数，得到最后结果0-9
model.add(Dense(num_classes, activation='softmax'))
# 模型使用交叉熵损失函数，最优化方法选用Adadelta
model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# 训练过程
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))
model.save('../files/my_model.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
