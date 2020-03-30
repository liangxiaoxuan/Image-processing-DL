mport numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Model
from numpy import *
from models.lenet import lenet_v1
from cleverhans.utils_tf import model_argmax
from sklearn.metrics import accuracy_score
from keras.layers import AveragePooling2D, Input,Activation
from diversity_metrics.metric_sheet import robustness,mean_variance,distance
from diversity_metrics.histogram_of_confidence import his_of_confi
from resources.common_corruption.common_corruption_cifar10 import common_corruption_loader
import pandas as pd


filepath = '/***.h5'  

# output = avg_logits

def independent_single(x_test_cc):

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32') / 255
    x_train = x_train.astype('float32') / 255
    input_shape = x_train.shape[1:]

    sess = tf.Session()
    keras.backend.set_session(sess)

    model_input = Input(shape=input_shape)  
    model_dic = {}
    model_out = []
    model_logits = []
    for i in range(3):
        model_dic[str(i)] = lenet_v1(X_input=model_input, num_classes=10)
        model_out.append(model_dic[str(i)][3])
        model_logits.append(model_dic[str(i)][2])

    model = Model(input=model_input, output=model_out)

    model.load_weights(filepath)
    pred = model(model_input)

    final_pred_list = []
    clean_pred_list = []
    confidence_list = []
    entropy_list = []
    for i in range(N_numbers):
        #sess.run(tf.global_variables_initializer())
        # f = sess.run(final_features, feed_dict={model_input: x_test_cc})   # features


        # confidence / cross_entropy
        # en = -np.sum(soft * np.log2(soft))
        # entropy_list.append(en)
        predictive_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=model_out[i], logits=model_logits[i])
        confidence = tf.math.reduce_max(model_out[i], axis=-1)
        pe_adv, conf_adv = sess.run([predictive_entropy, confidence], feed_dict={model_input: x_test_cc})
        entropy_list.append(pe_adv)
        confidence_list.append(conf_adv)

        # prediction
        final_pred = model_argmax(sess, model_input, pred[i], samples=x_test_cc)
        clean_pred = model_argmax(sess, model_input, pred[i], samples=x_test)
        final_pred_list.append(final_pred)
        clean_pred_list.append(clean_pred)

    return y_test, clean_pred_list, final_pred_list, entropy_list, confidence_list


if __name__ == '__main__':

    # # monolitic
    # r_ = []
    # acc_ = []
    # for s in range(2):
    #     for i in range(6):
    #         x_test_cc = common_corruption_loader(i, s)  # severity, noisetype
    #         y_test, clean_pred_list, final_pred_list, entropy_list, confidence_list = independent_single(x_test_cc)
    #         robust = robustness(y_test, clean_pred_list[0], final_pred_list[0])
    #         acc = accuracy_score(y_test, final_pred_list[0])
              r_.append(robust)
              acc_.append(acc)
     ## prediction (avg)
    each_severity_r = []
    each_severity_acc = []
    for s in range(2):  # noise
        for i in range(6):  # severity
            x_test_cc = common_corruption_loader(i, s)  # severity, noisetype
            y_test, clean_pred_list, final_pred_list, entropy_list, confidence_list = independent_single(x_test_cc)
            r_ = []
            acc_ = []
            for a in range(3):  # each net
                r = robustness(y_test, clean_pred_list[a], final_pred_list[a])
                acc = accuracy_score(final_pred_list[a], y_test)

                r_.append(r)
                acc_.append(acc)
                print(r_)
                print(acc_)

        
        
        
        
        
        
