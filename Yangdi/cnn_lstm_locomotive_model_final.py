#--coding=utf-8--
import tensorflow as tf
import pandas as pd
import numpy as np
#from sklearn.preprocessing import scale
#import model_constant as mc
import pdb
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
# model related
#------------------------------------------------

#得到权重变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1) #截断的随机分布输出，保证生成值在均值附近
    return tf.Variable(initial)

#得到偏置变量
def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

#CNN related
#------------------------------------------------

#num_channels：可理解为当前层的feature map的个数（filter个数）
#depth：是将要得到的feature map个数（filter个数）
def apply_conv(x,kernel_height,kernel_width,num_channels,depth):
    weights = weight_variable([kernel_height, kernel_width, num_channels, depth])
    biases = bias_variable([depth])
    #relu:计算激活函数，即max(features, 0)
    return weights, biases, tf.nn.relu(tf.add(tf.nn.conv2d(x, weights,[1,1,1,1],padding="VALID"),biases)) 

#max pool
#stride_size：步幅大小，应用max后向后滑动的元素个数    
def apply_max_pool(x,kernel_height,kernel_width,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1], strides=[1, 1, stride_size, 1], padding = "VALID")

#data related
#------------------------------------------------

#滑动窗口
def windows(nrows, size): #（数据行数，窗口打下）
    start,step = 0, 2     # 窗口开始位置，窗口滑动步长                                                                               
    while start+size < nrows:
        yield start, start + size #返回窗口的起始位置
        start += step

#数据分段
def segment_signal(features,labels, vars_size=24, window_size = 15): #（特征数据，数据对应标签，窗口大小默认15）
    segments = np.empty((0,window_size)) #返回未初始化的shape形状数据，此处相当于声明了个15列的变量
    segment_labels = np.empty((0,labels.shape[-1])) #此处相当于申请了变量
    nrows = len(features)-1 # have to save 1 step as prediction value
    for (start, end) in windows(nrows,window_size): #从生成器返回所有的窗口
        if(len(features[start:end]) == window_size):
            segment = features[start:end].T  #转置，得到24 x 15，相当于一个属性一行 
            #pdb.set_trace()
            label = labels[end:end+1]
            segments = np.vstack([segments,segment]) #垂直堆叠
            segment_labels = np.vstack([segment_labels,label]) #一行数据
    #TODO
    segments = segments.reshape(-1,vars_size,window_size,1) #batch_size不用管所以设为-1，channel也设为1 
    segment_labels = segment_labels.reshape(-1,labels.shape[-1]) #获得一列，行数根据实际展开
    return segments,segment_labels

class DataSet:
    train_x = np.empty(0)
    train_y = np.empty(0)
    test_x = np.empty(0) 
    test_y = np.empty(0) 

#根据比例随机划分训练集合测试集
#每个segement间是由时间关系的
#return Dictionary
def random_train_test(segments, labels, rate=0.7):
    train_test_split = np.random.rand(len(segments)) < rate
    #train dataset
    ds = DataSet()
    ds.train_x = segments[train_test_split]
    ds.train_y = labels[train_test_split]
    #test dataset
    ds.test_x = segments[~train_test_split]
    ds.test_y = labels[~train_test_split]
    return ds

#按照时序划分集合
def seq_train_test(segments, labels, rate=0.7):
    #pdb.set_trace()
    length = segments.shape[0]
    split_index = int(round(length*rate))
    ds = DataSet()
    #train dataset
    ds.train_x = segments[0:split_index,...]
    ds.train_y = labels[0:split_index,...]
    #test dataset 
    ds.test_x = segments[split_index:,...]
    ds.test_y = labels[split_index:,...]
    return ds

import helpFunctions as hf
#return Dictionary

#help function related
#---------------------------------------------
def cnt_mse(ori, pre):
    return tf.reduce_mean(tf.square(pre - ori))

def cnt_mse2(ori, pre):
    return ((pre-ori)**2).mean(axis=None)

def cnt_mse3(ori, pre, ax):
    return ((pre-ori)**2).mean(axis=ax)

def cnt_accuracy3(ori, pre, ax):
    tmp = [abs(pre[i]-ori[i])/ori[i] if ori[i].all!=0 else 0 for i in range(len(ori))]
    tmp = np.array(tmp)
    return (1-tmp.mean(axis=ax))*100

def cnt_accuracy2(ori, pre):
    tmp = [abs(pre[i]-ori[i])/ori[i] if ori[i].all!=0 else 0 for i in range(len(ori))]
    tmp = np.array(tmp)
    return (1-np.mean(tmp))*100

def cnt_accuracy(ori, pre):
    #pdb.set_trace()
    tmp = [abs(pre[i]-ori[i])/ori[i] if ori[i].all!=0 else 0 for i in range(len(ori))]
    #tmp = np.where(ori==0, 0, abs(pre-ori)/ori)
    #pdb.set_trace()
    tmp = np.array(tmp)
    accuracy = (1-np.mean(tmp))*100
    return tf.convert_to_tensor(accuracy)

def format_time_diff(time_diff):
    return time_diff.seconds/60, time_diff.seconds%60

import time
def get_version(pointedVersion=''):
    if pointedVersion=='':
        return time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    else: 
        return pointedVersion

def prepare_locomotive_data(self, filename, vars_size, window_size, prepared_data_file_name, split_type='random', rate=0.7):
    #read csv
    ds = DataSet() 
    try:
        ds = hf.readdic(prepared_data_file_name)
    except IOError:
        data = pd.read_csv(filename)
        features = np.array(data.as_matrix())
        labels = features[:,0:self.ct.num_output]
        '''
        lag = 1
        labels = data.iloc[:,0:6].shift(lag)
        #remove shifted Nan
        features = features[0:lag,:]
        labels = labels.iloc[0:lag,:]
        '''
        #segement data, 1 segement means one 'picture'
        segments, labels = segment_signal(features, labels, vars_size, window_size)
        #get random train and test set
        if split_type=='random':
            ds = random_train_test(segments, labels, rate)
        else:
            ds = seq_train_test(segments, labels, rate)
        #save to file
        hf.savedic(prepared_data_file_name, ds)
    return ds

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
def plot_mses(mses, fig_name):
    #n = len(mses)
    xmajorLocator   = MultipleLocator(10) #将x主刻度标签设置为n/10的倍数,即总共有10大格
    xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
    xminorLocator   = MultipleLocator(2) #将x轴次刻度标签设置为2的倍数，没大格有10/2小格
    '''
    ymajorLocator   = MultipleLocator(0.5) #将y轴主刻度标签设置为0.5的倍数
    ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式
    yminorLocator   = MultipleLocator(0.1) #将此y轴次刻度标签设置为0.1的倍数
    '''
    ax = plt.subplot(111)
    plt.plot(mses,'.')
    
    plt.title('MSEs of Training Epoches')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')

    #设置主刻度标签的位置,标签文本的格式
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    '''
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    '''
    #显示次刻度标签的位置,没有标签文本
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.grid(False, which='major') #x坐标轴的网格使用主刻度
    '''
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度
    '''
    #plt.show()
    plt.savefig(fig_name)

#---------------------------------------------
import datetime
class CNN_LSTM_MODEL:
    def __init__(self, version, filename, data_title,ct):
        self.version = version
        self.filename = filename
        self.data_title = data_title

        #ct = ModelConstants()
        self.ct = ct
        #定义：输入层
        self.X = X = tf.placeholder(tf.float32, shape=[None,ct.input_height,ct.input_width,1])
        self.Y = Y = tf.placeholder(tf.float32, shape=[None,ct.num_output])

        #定义：CNN层
        '''
        ck1_width = 4 #ct.input_height*
        pk1_width = 2 #1*
        ck2_width = 3 #1*
        pk2_width = 2 #1* 
        '''
        #第一层
        self.w1, self.b1, self.c1 = apply_conv(X, kernel_height = ct.input_height, kernel_width = 4, num_channels = 1, depth = 8)
        self.p1 = apply_max_pool(self.c1,kernel_height = 1, kernel_width = 2, stride_size = 2)
        #第二层
        self.w2, self.b2, self.c2 = apply_conv(self.p1, kernel_height = 1, kernel_width = 3, num_channels = 8, depth = 14)
        self.p2 = p2 = apply_max_pool(self.c2,kernel_height = 1, kernel_width = 2, stride_size = 2)

        #定义：LSTM层
        shape = p2.get_shape().as_list()
        time_step = shape[2]
        num_input = shape[3]
        lstm_input = tf.reshape(p2, [-1, time_step, num_input])
        lstm_input = tf.unstack(lstm_input, time_step, 1)
        self.lstm_w = lstm_w = weight_variable([ct.num_hidden_1, ct.num_output])
        self.lstm_b = lstm_b = bias_variable([ct.num_output])
        self.lstm_cell = lstm_cell = rnn.BasicLSTMCell(ct.num_hidden_1, forget_bias=1.0)
        if ct.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=ct.keep_prob)
        lstm_out, states = tf.nn.static_rnn(lstm_cell, lstm_input, dtype=tf.float32)
        self.y_ = y_ = tf.matmul(lstm_out[-1], lstm_w)+lstm_b

        '''
        #定义：全连接层
        #全连接输入
        shape = p2.get_shape().as_list()
        flat = tf.reshape(p2, [-1, shape[1] * shape[2] * shape[3]])#横向展开
        #隐藏层参数
        f_weights = weight_variable([shape[1] * shape[2] * shape[3], ct.num_hidden])
        f_biases = bias_variable([ct.num_hidden])
        f = tf.nn.tanh(tf.add(tf.matmul(flat, f_weights),f_biases))

        #定义：输出层
        out_weights = weight_variable([ct.num_hidden, 1])
        out_biases = bias_variable([ct.num_output])
        y_ = tf.add(tf.matmul(f, out_weights),out_biases)
        '''

        #定义：模型优化
        self.cost_function = cost_function = tf.reduce_mean(tf.square(y_- Y))#损失函数
        self.optimizer = tf.train.AdamOptimizer(ct.learning_rate).minimize(cost_function)#梯度下降
        
        self.train_log_file_name = data_title+'_train_log_file_'+version+'.txt'

        self.saver = tf.train.Saver()
        
        self.session = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def train(self, train_x, train_y, model_file_name):
        self.ct.total_batches = train_x.shape[0]
        #Train
        #----------------------------------------------
        session = self.session

        f = open(self.train_log_file_name, 'a+')
        time_tag = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        f.write(time_tag+"\n")
        f.write("-------------------\n")
        f.write('originial data file name: '+filename+"\n")
        f.write('train data size: %d\n'%(train_x.shape[0]))
        f.write('input_height %d\n'%(self.ct.input_height))
        f.write('input_width %d\n'%(self.ct.input_width))
        f.write('lstm num_hidden_1 %d\n'%(self.ct.num_hidden_1))
        f.write('learning_rate %d\n'%(self.ct.learning_rate))
        f.write('batch_size %d\n'%(self.ct.batch_size))

        #训练
        print '\nStart Training...'
        print(time_tag)
        time_train_start = datetime.datetime.now()
        tr_mses = list()
        for epoch in range(self.ct.training_epochs):
            for b in range(self.ct.total_batches):
                offset = (b * self.ct.batch_size) % (train_x.shape[0] - self.ct.batch_size)
                batch_x = train_x[offset:(offset + self.ct.batch_size), :, :, :]
                batch_y = train_y[offset:(offset + self.ct.batch_size),:]
                #pdb.set_trace()
                _, c = session.run([self.optimizer, self.cost_function],feed_dict={self.X: batch_x, self.Y : batch_y})
                #pdb.set_trace()
                #print b
            #MSE
            p_tr = session.run(self.y_, feed_dict={self.X: train_x})
            #pdb.set_trace()#iTODO
            tr_mse = cnt_mse(train_y[:,0:self.ct.predict_vars_num], p_tr[:,0:self.ct.predict_vars_num])
            tr_mses.append(tr_mse.eval())
            #Acuuracy Percetage
            accuracy = cnt_accuracy(train_y[:,0:self.ct.predict_vars_num], p_tr[:,0:self.ct.predict_vars_num])
            if (epoch+1)%self.ct.epoch_out_size == 0:
                f.write("\nepoch:%d mse:%.4f accuracy: %.2f"%(epoch+1,session.run(tr_mse), session.run(accuracy)))
                print("epoch:%d mse:%.4f accuracy: %.2f"%(epoch+1,session.run(tr_mse),session.run(accuracy)))
            # once the mse decreased, stop training
        fig_version = time.strftime('%Y%m%d%H%M', time.strptime(time_tag, '%Y-%m-%d %H:%M:%S'))
        mse_fig_name = self.data_title+'_fig_mses_'+self.version+'_'+fig_version+'.png'
        plot_mses(tr_mses, mse_fig_name)
        time_train_end = datetime.datetime.now()
        f.write('\nTraining time: %d m %d s'%(format_time_diff(time_train_end-time_train_start)))
        print('Training time: %d m %d s'%(format_time_diff(time_train_end-time_train_start)))
        
        self.saver.save(session, model_file_name)

        f.write("\n\n")
        f.close()
        #session.close()
        return format_time_diff(time_train_end-time_train_start)
    
    def test(self, test_x, test_y, model_file_name):
        #pdb.set_trace()
        session = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.saver.restore(session, model_file_name)  
        f = open(self.train_log_file_name, 'a+')
        #测试
        print '\nStart Testing...'
        time_test_start = datetime.datetime.now()
        #MSE
        p_ts = session.run(self.y_, feed_dict={self.X: test_x})
        ts_mse = cnt_mse(test_y[:,0:self.ct.predict_vars_num], p_ts[:,0:self.ct.predict_vars_num])
        f.write("\n\nTest set MSE: %.4f" % session.run(ts_mse))
        print("Test set MSE: %.4f" % session.run(ts_mse))
        #Accuracy percentage
        accuracy = cnt_accuracy(test_y[:,0:self.ct.predict_vars_num], p_ts[:,0:self.ct.predict_vars_num])
        f.write("\naccuracy: %.4f" % session.run(accuracy))
        print("accuracy: %.4f" % session.run(accuracy))
        time_test_end = datetime.datetime.now()
        time_diff = time_test_end - time_test_start
        f.write('\nTesting  time: %d m %d s'%(format_time_diff(time_diff)))
        print('Testing  time: %d m %d s'%(format_time_diff(time_diff)))
        
        test_mse = session.run(ts_mse)
        test_acc = session.run(accuracy)

        f.write("\n\n")
        f.close()
        session.close()
        
        return test_mse, test_acc
    
    #split input into train/test sets
    #train the model
    def train_and_test(self, model_file_name):
        prepared_data_file_name = data_title+'_prepared_data_file_'+version+'.pkl'
        ds = prepare_locomotive_data(self, filename, self.ct.input_height, self.ct.input_width, prepared_data_file_name, split_type='seq')
        train_time = self.train(ds.train_x, ds.train_y, model_file_name)
        test_mse, test_acc = self.test(ds.test_x, ds.test_y, model_file_name)
        return train_time, test_mse, test_acc
    
    #use other locomotive's data as input
    #use this function to test the generality of the model
    def model_test(self, test_file_name, model_file_name):
        f = open(self.train_log_file_name, 'a+')
        time_tag = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        f.write(time_tag+"\n")
        f.write("-------------------")
        f.write('\nModel Test File: '+test_file_name)
        data = pd.read_csv(test_file_name)
        features = np.array(data.as_matrix())
        labels = features[:,0:self.ct.num_output] 
        segments, labels = segment_signal(features, labels, self.ct.input_height, self.ct.input_width) 
        f.write('\nTest data size: %d'%segments.shape[0])
        print time_tag+'\n'
        f.close()
        self.test(segments, labels, model_file_name)
    
    #use single splited data 'picture' as input 
    def one_step_prediction(self, im, model_file_name):
        session = tf.InteractiveSession(model.y_,)
        tf.global_variables_initializer().run()
        self.saver.restore(session, model_file_name)
        pre = self.y_.eval(feed_dict={self.X: im})
        session.close()
        return pre
    
    #given the previours data 'picture'
    #predict one step and add back to form a new 'picture'
    #recursively do the prediction
    def multi_step_prediction(self, im, model_file_name, step, outs):
        session = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.saver.restore(session, model_file_name)
        pres = np.empty((0,self.ct.num_output)) 
        for i in range(step):
            #add predicted one step to form new input
            pre = self.y_.eval(feed_dict={self.X: im})
            pres = np.vstack([pres,pre])

            pre = pre.reshape(1,pre.shape[-1],1,1)
            im = np.append(im, pre, axis=2)
            im = im[:,:,0:-1,:]
        session.close()
        
        return cnt_mse3(outs[:,0:6],pres[:,0:6],1),cnt_accuracy3(outs[:,0:6],pres[:,0:6],1)

def multi_step_prediction2(im, model, step, outs):
    pres = np.empty((0,model.ct.num_output))
    for i in range(step):
        pre = model.session.run(model.y_,feed_dict={model.X: im})
        pres = np.vstack([pres,pre])
        pre = pre.reshape(1,pre.shape[-1],1,1)
        im = np.append(im, pre, axis=2)
        im = im[:,:,0:-1,:]
    return cnt_mse3(outs[:,0:6],pres[:,0:6],1),cnt_accuracy3(outs[:,0:6],pres[:,0:6],1)

#TODO
class ModelConstants:
    batch_size = 2
    num_hidden_1 = 12
    #num_hidden_2 = 100
    learning_rate = 0.001
    training_epochs = 100
    input_height =13 
    input_width = 15
    num_output = 13
    keep_prob = 0.8 #dropout率，每批数据输入时神经网络中的每个单元会以1-keep_prob的概率不工作，可以防止过拟合
    predict_vars_num = 6
    epoch_out_size =  10

    total_batches = 0 

def split_data(filename):
    ct = ModelConstants()
    data = pd.read_csv(filename)
    features = np.array(data.as_matrix())
    labels = features[:,0:ct.num_output] 
    segments, labels = segment_signal(features, labels, model.ct.input_height, model.ct.input_width)
    #print 'window size:%d'%model.ct.input_width
    return segments, labels

def split_data2(filename, ct):
    data = pd.read_csv(filename)
    features = np.array(data.as_matrix())
    labels = features[:,0:ct.num_output] 
    segments, labels = segment_signal(features, labels, ct.input_height, ct.input_width)
    print 'window size:%d'%ct.input_height
    return segments, labels
   

#TODO
filename = ""
#filename = '~/AXIS_data/163_0168-1~10/1_163_168_2016-04-30_ip_10_axis_6.csv'
#filename = '~/AXIS_data/163_0168-1~10/2_163_168_2016-04-18_ip_10_axis_1.csv'
#filename = '~/AXIS_data/163_0168-1~10/3_163_168_2016-04-21_ip_10_axis_1.csv'
#filename = '~/AXIS_data/163_0168-1~10/4_163_168_2016-04-20_ip_10_axis_6.csv'
#filename = '~/AXIS_data/163_0168-1~10/5_163_168_2016-04-11_ip_10_axis_6.csv'
#filename = '~/AXIS_data/163_0168-1~10/6_163_168_2016-08-05_ip_10_axis_6.csv'
#filename = '~/AXIS_data/163_0168-1~10/7_163_168_2016-04-22_ip_10_axis_1.csv'
#filename = '~/AXIS_data/163_0168-1~10/8_163_168_2016-04-23_ip_10_axis_1.csv'
#filename = '~/AXIS_data/163_0168-1~10/9_163_168_2016-04-19_ip_10_axis_1.csv'
#filename = '~/AXIS_data/163_0168-1~10/10_163_168_2016-04-16_ip_10_axis_1.csv'

#filename = '~/AXIS_data/同车型线路/1-31_233_0790_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv'
#filename = '~/AXIS_data/同车型线路/2-9_233_0192_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv'
#filename = '~/AXIS_data/同车型线路/3-62_233_0520_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv'
#filename = '~/AXIS_data/同车型线路/4-53_233_0609_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv'
#filename = '~/AXIS_data/同车型线路/5-60_233_0500_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv'
#filename = '~/AXIS_data/同车型线路/6-4_233_0134_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv'
#filename = '~/AXIS_data/同车型线路/7-13_233_0394_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv'
#filename = '~/AXIS_data/同车型线路/8-71_233_0141_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv'

filelist_1 = [
        '~/AXIS_data/同车型线路/1-31_233_0790_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv',
        '~/AXIS_data/同车型线路/2-9_233_0192_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv',
        '~/AXIS_data/同车型线路/3-62_233_0520_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv',
        '~/AXIS_data/同车型线路/4-53_233_0609_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv',
        '~/AXIS_data/同车型线路/5-60_233_0500_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv',
        '~/AXIS_data/同车型线路/6-4_233_0134_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv',
        '~/AXIS_data/同车型线路/7-13_233_0394_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv',
        '~/AXIS_data/同车型线路/8-71_233_0141_CC-66666_2016-05-20_merge_r_ri_10_axis_1.csv'
        ]
filelist_2 = [ 
        '~/AXIS_data/163_0168-1~10/1_163_168_2016-04-30_ip_10_axis_1.csv',
        '~/AXIS_data/163_0168-1~10/2_163_168_2016-04-18_ip_10_axis_1.csv',
        '~/AXIS_data/163_0168-1~10/3_163_168_2016-04-21_ip_10_axis_1.csv',
        '~/AXIS_data/163_0168-1~10/4_163_168_2016-04-20_ip_10_axis_1.csv',
        '~/AXIS_data/163_0168-1~10/5_163_168_2016-04-11_ip_10_axis_1.csv',
        '~/AXIS_data/163_0168-1~10/6_163_168_2016-08-05_ip_10_axis_1.csv',
        '~/AXIS_data/163_0168-1~10/7_163_168_2016-04-22_ip_10_axis_1.csv',
        '~/AXIS_data/163_0168-1~10/8_163_168_2016-04-23_ip_10_axis_1.csv'
        ]

version = get_version('20180119_1993')#TODO 
data_title = 'locmotive'

# Train the model
ct = ModelConstants()
model = CNN_LSTM_MODEL(version, filename, data_title, ct)

switch = 33333 #TODO  
model_file_name = data_title+'_model_'+version+'.ckpt'
if switch==1:
    #Chose the right learning parameter
    tmp = [0.0001,0.001,0.01,0.1]
    parameter_filename = 'parameter_lr_'+version+'.csv'
    f = open(parameter_filename, 'w')
    f.write('lr,train_time,test_mse,test_accuracy\n')
    filename = '~/AXIS_data/163_0168-1~10/2_163_168_2016-04-18_ip_10_axis_1.csv'
    segments, labels = split_data(filename)
    #pdb.set_trace()
    ds = seq_train_test(segments, labels)
    for i in range(len(tmp)):
        model.ct.learning_rate = tmp[i]
        train_time = model.train(ds.train_x, ds.train_y, model_file_name)
        #pdb.set_trace()
        test_mse, test_acc = model.test(ds.test_x, ds.test_y, model_file_name)
        f.write('%f,%f,%d,%d,%f'%(tmp[i],train_time[0],train_time[1],test_mse,test_acc))
        #pdb.set_trace()
    f.close()
elif switch==11:
    #plot original data
    data = pd.read_csv(filename)
    features = np.array(data.as_matrix())
    plt.figure(figsize=(20,1*13))
    for i in range(13):
        plt.subplot(13,1,i+1)
        plt.plot(features[:,i])
    plt.show()
    #pdb.set_trace()
elif switch==2:
    #Chose the right window_size 
    parameter_filename = 'output_cnn_lstm_files_20180102_1006.csv' #'output_cnn_lstm_'+version+'_'+filename.split('/')[-1]
    f = open(parameter_filename, 'a+')
    #f.write('wsize,train_time,test_mse,test_accuracy\n')
    #filename = '~/AXIS_data/163_0168-1~10/2_163_168_2016-04-18_ip_10_axis_1.csv'
    segments, labels = split_data(filename)
    print filename
    #pdb.set_trace()
    ds = seq_train_test(segments, labels)
    train_time = model.train(ds.train_x, ds.train_y, model_file_name)
    test_mse, test_acc = model.test(ds.test_x, ds.test_y, model_file_name)
    f.write('%s,%d m %d s,%.4f,%.2f\n'%(filename.split('/')[-1],train_time[0],train_time[1],test_mse,test_acc))
    f.close()
    '''
    for i in range(11,31):
        ct = ModelConstants()
        ct.input_width = i
        model = CNN_LSTM_MODEL(version, filename, data_title, ct)
        segments, labels = split_data2(filename,ct)
        ds = seq_train_test(segments, labels)
        train_time = model.train(ds.train_x, ds.train_y, model_file_name)
        #pdb.set_trace()
        test_mse, test_acc = model.test(ds.test_x, ds.test_y, model_file_name)
        f.write('%f,%d m %d s,%f,%f\n'%(i,train_time[0],train_time[1],test_mse,test_acc))
        #pdb.set_trace()
    '''
elif switch==3: #多个文件-单步预测
    filelist = filelist_2
    out_filename = 'output_cnn_lstm_mulfile2_'+version+'.csv'
    f = open(out_filename, 'a+')
    
    train_x = np.empty((0,ct.input_height,ct.input_width,1))
    train_y = np.empty((0,ct.num_output))
    train_index = [1,2,3,4,5] #TODO
    train_str = str(train_index[0])
    for i in train_index:
        if i<>train_index[0]:
            train_str = train_str+'_'+str(i)
        tmp_x, tmp_y = split_data(filelist[i-1])
        print filelist[i-1]
        train_x = np.vstack([train_x,tmp_x])
        train_y = np.vstack([train_y,tmp_y])
    train_time = model.train(train_x, train_y, model_file_name)
    f.write('%s,%d m %d s\n'%(train_str,train_time[0],train_time[1]))
    
    for i in range(len(filelist)):
        test_x, test_y = split_data(filelist[i])
        mse, acc = model.test(test_x, test_y, model_file_name)
        f.write('%.4f,%.2f\n'%(mse,acc))        
    f.write('\n')
    f.close()
elif switch==33: #多个文件-多步预测
    filelist = filelist_1 #TODO
    mfile = 1
    train_index = [2] #TODO
    pre_step = 180 #TODO
    out_filename = 'output_cnn_lstm_mfile'+str(mfile)+'_mstep'+str(pre_step)+'_'+version+'_2.csv'
    print(out_filename)

    train_x = np.empty((0,ct.input_height,ct.input_width,1))
    train_y = np.empty((0,ct.num_output))
    train_str = str(train_index[0])
    for i in train_index:
        if i<>train_index[0]:
            train_str = train_str+'_'+str(i)
        tmp_x, tmp_y = split_data(filelist[i-1])
        print filelist[i-1]
        train_x = np.vstack([train_x,tmp_x])
        train_y = np.vstack([train_y,tmp_y])
    train_time = model.train(train_x, train_y, model_file_name)
    print('%s,%d m %d s'%(train_str,train_time[0],train_time[1]))
   
    mses = np.empty((0,pre_step))
    accs = np.empty((0,pre_step))
    for i in range(len(filelist)):
        print('test: %s'%filelist[i])
        test_x, test_y = split_data(filelist[i])
        k = 0
        print(len(test_x))
        while k+pre_step <= len(test_x):
            outs = test_y[k:(k+pre_step)]
            mse, acc = multi_step_prediction2(test_x[k:(k+1)], model, pre_step, outs)
            mses = np.vstack([mses, mse])
            accs = np.vstack([accs, acc])
            k = k+1
    mse = mses.mean(axis=0)
    acc = accs.mean(axis=0)
    #out_filename = 'output_cnn_lstm_mfile'+str(mfile)+'_mstep'+str(pre_step)+'_'+version+'_2.csv'
    print(out_filename)
    f = open(out_filename, 'a+')
    f.write('%s\n'%train_str)
    for i in range(pre_step):
        f.write('%.4f,%.2f\n'%(mse[i],acc[i]))
    f.write('\n')
    f.close()


elif switch==333:
    #Chose the right hidden_node 
    parameter_filename = 'parameter_hnum_'+version+'.csv'
    f = open(parameter_filename, 'w')
    f.write('hnum,train_time,test_mse,test_accuracy\n')
    #filename = '~/AXIS_data/163_0168-1~10/2_163_168_2016-04-18_ip_10_axis_1.csv'
    segments, labels = split_data(filename)
    ds = seq_train_test(segments, labels)
    tmp = range(6,16)
    b = [20,30,40,50,100,200]
    tmp.extend(b)
    for i in range(len(tmp)):
        model.ct.num_hidden_1 = tmp[i]
        train_time = model.train(ds.train_x, ds.train_y, model_file_name)
        #pdb.set_trace()
        test_mse, test_acc = model.test(ds.test_x, ds.test_y, model_file_name)
        f.write('%f,%d m %d s,%f,%f\n'%(tmp[i],train_time[0],train_time[1],test_mse,test_acc))
        #pdb.set_trace()
    f.close()
elif switch==4:
    #Chose the right dropout rate keep_prob 
    parameter_filename = 'parameter_drop_'+version+'.csv'
    f = open(parameter_filename, 'w')
    f.write('drop,train_time,test_mse,test_accuracy\n')
    #filename = '~/AXIS_data/163_0168-1~10/2_163_168_2016-04-18_ip_10_axis_1.csv'
    segments, labels = split_data(filename)
    ds = seq_train_test(segments, labels)
    tmp = np.arange(0.5,1.05,0.1)
    for i in range(len(tmp)):
        model.ct.keep_prob = tmp[i]
        train_time = model.train(ds.train_x, ds.train_y, model_file_name)
        #pdb.set_trace()
        test_mse, test_acc = model.test(ds.test_x, ds.test_y, model_file_name)
        f.write('%f,%d m %d s,%f,%f\n'%(tmp[i],train_time[0],train_time[1],test_mse,test_acc))
        #pdb.set_trace()
    f.close()
elif switch==5:
    #Chose the right batch size 
    parameter_filename = 'parameter_batch_'+version+'.csv'
    f = open(parameter_filename, 'w')
    f.write('batch,train_time,test_mse,test_accuracy\n')
    #filename = '~/AXIS_data/163_0168-1~10/2_163_168_2016-04-18_ip_10_axis_1.csv'
    segments, labels = split_data(filename)
    ds = seq_train_test(segments, labels)
    tmp = np.arange(1,11,1)
    for i in range(len(tmp)):
        model.ct.batch_size = tmp[i]
        train_time = model.train(ds.train_x, ds.train_y, model_file_name)
        #pdb.set_trace()
        test_mse, test_acc = model.test(ds.test_x, ds.test_y, model_file_name)
        f.write('%f,%d m %d s,%f,%f\n'%(tmp[i],train_time[0],train_time[1],test_mse,test_acc))
        #pdb.set_trace()
    f.close()
   
    version = get_version('')
    model_file_name = data_title+'_model_'+version+'.ckpt'
    #Chose the right epoches 
    parameter_filename = 'parameter_epoch_'+version+'.csv'
    f = open(parameter_filename, 'w')
    f.write('epoch,train_time,test_mse,test_accuracy\n')
    #filename = '~/AXIS_data/163_0168-1~10/2_163_168_2016-04-18_ip_10_axis_1.csv'
    segments, labels = split_data(filename)
    ds = seq_train_test(segments, labels)
    tmp = np.arange(10,110,10)
    for i in range(len(tmp)):
        model.ct.training_epochs = tmp[i]
        train_time = model.train(ds.train_x, ds.train_y, model_file_name)
        #pdb.set_trace()
        test_mse, test_acc = model.test(ds.test_x, ds.test_y, model_file_name)
        f.write('%f,%d m %d s,%f,%f\n'%(tmp[i],train_time[0],train_time[1],test_mse,test_acc))
        #pdb.set_trace()
    f.close()

elif switch==7:
    model.train_and_test(model_file_name)
elif switch==33:#single step prediction
    model.model_test(filename, model_file_name)
elif switch==44:#use as less data as possible to train and predict
    ins, outs = split_data(filename)
    s_train = 0
    e_train = s_train+600
    s_test = 600
    e_test = s_test+100
    model.train(ins[s_train:e_train], outs[s_train:e_train],model_file_name)
    model.test(ins[s_test:e_test], outs[s_test:e_test], model_file_name)  
    pdb.set_trace()
else:#multi_step prediction
    tag = 3 #TODO
    if tag==1:#单文件多步预测
        #Read data
        filelist = filelist_1 #TODO
        index = 8 #TODO
        filename = filelist[index-1] 
        print(filename)
        data = pd.read_csv(filename)
        features = np.array(data.as_matrix())
        labels = features[:,0:model.ct.num_output] 
        segments, labels = segment_signal(features, labels, model.ct.input_height, model.ct.input_width)
        ds = seq_train_test(segments, labels)
        #Train
        train_time = model.train(ds.train_x, ds.train_y, model_file_name)
        #Test
        pre_step = 180 #TODO
        mses = np.empty((0,pre_step))
        accs = np.empty((0,pre_step))
        times = np.empty((0,2))
        i = 0
        while i+pre_step <= len(ds.test_x):
            outs = ds.test_y[i:(i+pre_step)]
            time_start = datetime.datetime.now()
            mse, acc = model.multi_step_prediction(ds.test_x[i:(i+1)], model_file_name, pre_step, outs)
            time_end = datetime.datetime.now() 
            test_time = format_time_diff(time_end-time_start)
            times = np.vstack([times, test_time])
            mses = np.vstack([mses, mse])
            accs = np.vstack([accs, acc])
            i = i+1
        test_time = times.mean(axis=0)
        mse = mses.mean(axis=0)
        acc = accs.mean(axis=0)
        f_mse = open('output_cnn_lstm_sfile2_mstep'+str(pre_step)+'_'+version+'_mse.csv','a+')
        f_acc = open('output_cnn_lstm_sfile2_mstep'+str(pre_step)+'_'+version+'_acc.csv','a+')
        f_mse.write('%d,%d m %d s\n'%(index,test_time[0],test_time[1]))
        f_acc.write('%d,%d m %d s\n'%(index,test_time[0],test_time[1]))
        for i in range(pre_step):
            f_mse.write('%.4f\n'%mse[i])
            f_acc.write('%.2f\n'%acc[i])
        f_mse.write('\n')
        f_acc.write('\n')
        f_mse.close()
        f_acc.close()
    elif tag==2:
        filelist = filelist_2 #TODO
        mfile = 2 #TODO
        index = 1 #TODO
        train_step = (2*3600)/10 #TODO\
        pre_step = 1 #TODO
        out_filename = 'output_cnn_lstm_sfile'+str(mfile)+'_'+str(index)+'_tstep'+str(train_step)+'_pstep'+str(pre_step)+'_'+version+'.csv'
        filename = filelist[index-1]
        print(filename)
        print(out_filename)
        
        data = pd.read_csv(filename) 
        features = np.array(data.as_matrix())
        labels = features[:,0:model.ct.num_output]
        segments, labels = segment_signal(features, labels, model.ct.input_height, model.ct.input_width)
        train_x = segments[0:train_step]
        train_y = labels[0:train_step]
        test_x = segments[train_step:,...]
        test_y = labels[train_step:,...]
       
        model.train(train_x, train_y, model_file_name)

        mses = np.empty((0,pre_step))
        accs = np.empty((0,pre_step))
        k = 0
        while k+pre_step <= len(test_x):
            outs = test_y[k:(k+pre_step)]
            mse, acc = multi_step_prediction2(test_x[k:(k+1)], model, pre_step, outs)
            mses = np.vstack([mses, mse])
            accs = np.vstack([accs, acc])
            k = k+1
        f = open(out_filename, 'a+')
        f.write('mse,acc\n')
        for i in range(len(mses)):
            f.write('%.4f,%.2f\n'%(mses[i,0],accs[i,0]))
        f.write('\n')
        f.close()
    elif tag==3:#单文件-间隔训练
        filelist = filelist_2 #TODO
        mfile = 2 #TODO
        index = 2 #TODO
        train_step = int((0.5*3600)/10) #TODO
        pre_step = int((0.5*3600)/10) #TODO
        out_filename_1 = 'output_cnn_lstm_retrain_sfile'+str(mfile)+'_'+str(index)+'_tstep'+str(train_step)+'_pstep'+str(pre_step)+'_'+version+'_mse.csv'
        out_filename_2 = 'output_cnn_lstm_retrain_sfile'+str(mfile)+'_'+str(index)+'_tstep'+str(train_step)+'_pstep'+str(pre_step)+'_'+version+'_acc.csv'
        filename = filelist[index-1]
        print(filename)
        print(out_filename_1)
        
        data = pd.read_csv(filename)
        features = np.array(data.as_matrix())
        labels = features[:,0:model.ct.num_output]
        segments, labels = segment_signal(features, labels, model.ct.input_height, model.ct.input_width)
        
        retrain_times = int((len(segments)-pre_step)/train_step)
       
        msess = np.empty((0,pre_step))
        accss = np.empty((0,pre_step))
        for i in range(retrain_times-1):
            s = i*train_step
            e = (i+1)*train_step
            train_x = segments[s:e]
            train_y = labels[s:e]
            model.train(train_x, train_y, model_file_name)
            
            mses = np.empty((0,pre_step))
            accs = np.empty((0,pre_step))
            for k in range(train_step):
                outs = labels[(e+k):(e+k+pre_step)]
                mse, acc = multi_step_prediction2(segments[(e+k):(e+k+1)], model, pre_step, outs)
                mses = np.vstack([mses, mse])
                accs = np.vstack([accs, acc])
            mse = mses.mean(axis=0)
            acc = accs.mean(axis=0)
            msess = np.vstack([msess, mse])
            accss = np.vstack([accss, acc])

        f_1 = open(out_filename_1, 'a+')
        f_2 = open(out_filename_2, 'a+')
        for i in range(msess.shape[0]):
            f_1.write('%.2f'%msess[i,0])
            f_2.write('%.2f'%accss[i,0]) 
            for j in range(1,msess.shape[1]):
                f_1.write(',%.2f'%msess[i,j])
                f_2.write(',%.2f'%accss[i,j])
            f_1.write('\n')
            f_2.write('\n')
        f_1.close()
        f_2.close() 
    else:
        pdb.set_trace()
