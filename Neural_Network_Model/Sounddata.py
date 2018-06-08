

import os
import os.path
import math

import numpy as np
import tensorflow as tf

import Sounddata_input

#%%

BATCH_SIZE = 20
learning_rate = 0.001
MAX_STEP = 40000
keep_prob=0.5

#%%

def inference(images):
    '''
    Args:
        images: 4D tensor [batch_size, img_width, img_height, img_channel]
    Notes:
        In each conv layer, the kernel size is:
        [kernel_size, kernel_size, number of input channels, number of output channels].
        number of input channels are from previuous layer, if previous layer is THE input
        layer, number of input channels should be image's channels.
        

    '''
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [5, 5, 1, 32],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32)) 
        biases = tf.get_variable('biases', 
                                 shape=[32],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
        #drop1 = tf.nn.dropout(conv1, keep_prob)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)

    #pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')
        drop2=tf.nn.dropout(norm1,keep_prob)


    #conv2
    with tf.variable_scope('conv2') as scope:
        weights2 = tf.get_variable('weights',
                                  shape=[5,5,32,64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(drop2, weights2, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
        #drop3 = tf.nn.dropout(conv2, keep_prob)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights2)

    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
       norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                         beta=0.75,name='norm2')
       pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                              padding='SAME',name='pooling2')
       drop4 = tf.nn.dropout(pool2, keep_prob)
    
    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(drop4, shape=[BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights3 = tf.get_variable('weights',
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights3) + biases, name=scope.name)
        #drop4 = tf.nn.dropout(local3, keep_prob)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights3)

    #local4
    with tf.variable_scope('local4') as scope:
        weights4 = tf.get_variable('weights',
                                  shape=[128,64],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights4) + biases, name='local4')
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights4)

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[64, 4],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[4],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    
    return softmax_linear

#%%

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        
        labels = tf.cast(labels, tf.int64)
        
        # to use this loss fuction, one-hot encoding is needed!
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')

                        
#        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
#                        (logits=logits, labels=labels, name='xentropy_per_example')
        regularizer=tf.contrib.layers.l2_regularizer(scale=5.0/30000)
        reg_term=tf.contrib.layers.apply_regularization(regularizer,weights_list=None)

        loss =tf.reduce_mean(cross_entropy, name='loss')#+reg_term)
        tf.summary.scalar(scope.name+'/loss', loss)
        
    return loss

def evaluate():
    with tf.Graph().as_default():

        #log_dir = 'C:\\Users\\Ruoyu\\PycharmProjects\\Cifar-10\\logstrain'
        log_dir = 'F:\\cut_depth_data\\logtrain_k5_2c2m_drop0.5_40000'
        #test_dir = 'C:\\Users\\Ruoyu\\PycharmProjects\\Cifar-10\\cifar-10-batches-bin'
        test_dir='F:\\cut_depth_data\\test'
        n_test = 3140


        # reading test data
        images, labels = Sounddata_input.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size= BATCH_SIZE,
                                                    shuffle=False)

        logits = inference(images)
        #labels=tf.cast(labels, tf.int32)
        labels = tf.argmax(labels, 1)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)

            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
                print('precision = %.3f' % precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        labels = tf.argmax(labels, 1)  # one_hot decode
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

def train_val():
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    #keep_prob=tf.placeholder(tf.float32)
    # starter_learning_rate = 0.001
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, my_global_step,
    #                                            4000, 0.2, staircase=True)# changing LR
    data_dir = 'F:\\cut_depth_data'
    log_train_dir = 'F:\\cut_depth_data\\logtrain_k5_2c2m_drop0.5_40000'
    log_val_dir = 'F:\\cut_depth_data\\logvaln__k5_2c2m_drop0.5_40000'
    #val_dir = 'C:\\Users\\Ruoyu\\PycharmProjects\\Cifar-10\\cifar-10-batches-bin'


    images, labels = Sounddata_input.read_cifar10(data_dir=data_dir,
                                                      is_train=True,
                                                    batch_size=BATCH_SIZE,
                                                       shuffle=True)
    images_val,labels_val=Sounddata_input.read_cifar10(data_dir=data_dir,
                                                           is_train=False,
                                                         batch_size=BATCH_SIZE,
                                                            shuffle=False)


    logits = inference(images)
    #logits=inference(images,keep_prob=0.5)
    loss = losses(logits, labels)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=my_global_step)
    accuracy = evaluation(logits, labels)

    x=tf.placeholder(tf.float32,shape=[BATCH_SIZE,150,150,1])
    y_=tf.placeholder(tf.int32, shape=[BATCH_SIZE,4])
    #keep_prob=tf.placeholder(tf.float32)



    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_writer = tf.summary.FileWriter(log_train_dir, sess.graph)
    val_writer=tf.summary.FileWriter(log_val_dir,sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            tra_image,tra_label=sess.run([images,labels])
            _, tra_loss,tra_acc = sess.run([train_op, loss, accuracy],
                                             feed_dict={x:tra_image,y_:tra_label})
            # _, loss_value = sess.run([train_op, loss])


            if step % 50 == 0:
                print('Step: %d, loss: %.4f,train accuracy = %.2f%%' %(step, tra_loss,tra_acc * 100.0))
                # print('Step: %d, loss: %.4f' %(step, loss_value))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)


            if step % 200 == 0:
                val_image,val_label = sess.run([images_val,labels_val])
                val_loss,val_acc = sess.run([loss,accuracy],
                                            feed_dict={x:val_image,y_:val_label})
                print('<  Step: %d, valid loss: %.4f,valid accuracy = %.2f%%  >' % (step, val_loss, val_acc * 100.0))
                summary_str = sess.run(summary_op)
                val_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(log_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

train_val()
#evaluate()


# Draw confusion matrix of classification result

'''''
import scipy.io as scio
def confusion_matrix():
      with tf.Graph().as_default():

          log_dir = 'F:\\cut_depth_data\\logtrain_k5_2c2m_drop0.5'
          test_dir = 'F:\\cut_depth_data\\test'
          n_test = 1800

          # reading test data
          images, labels = cifar10_input.read_cifar10(data_dir=test_dir,
                                                      is_train=False,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=False)

          logits = inference(images)
          labels = tf.argmax(labels, 1) # one_hot decode
          value,id=tf.nn.top_k(logits)
          predict=id
          # top_k_op = tf.nn.in_top_k(logits, labels, 1)
          saver = tf.train.Saver(tf.global_variables())

          with tf.Session() as sess:

              print("Reading checkpoints...")
              ckpt = tf.train.get_checkpoint_state(log_dir)
              if ckpt and ckpt.model_checkpoint_path:
                  global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                  saver.restore(sess, ckpt.model_checkpoint_path)
                  print('Loading success, global_step is %s' % global_step)
              else:
                  print('No checkpoint file found')
                  return

              coord = tf.train.Coordinator()
              threads = tf.train.start_queue_runners(sess=sess, coord=coord)

              try:
                  num_iter = int(math.ceil(n_test / BATCH_SIZE))
                  print(num_iter)
                  true_count = 0
                  total_sample_count = num_iter * BATCH_SIZE
                  step = 0
                  predict_label_all=np.zeros((20,1))
                  true_label_all=np.zeros((1,20))
                  while step < num_iter and not coord.should_stop():
                      predict_label,true_label=sess.run([predict,labels])
                      predict_label_all=np.hstack((predict_label_all,predict_label))
                      true_label_all=np.vstack((true_label_all,true_label))
                      # print(predict_label)
                      print(true_label)
                      # predictions = sess.run([top_k_op])
                      # true_count += np.sum(predictions)
                      step += 1
                      # precision = true_count / total_sample_count
                  # print('precision = %.3f' % precision)
                  return np.reshape(np.transpose(predict_label_all[:,1:]),(1,1800)),np.reshape(true_label_all[1:,:],(1,1800))
              except Exception as e:
                  coord.request_stop(e)
              finally:
                  coord.request_stop()
                  coord.join(threads)
# #
# #                 # %%
#
predict_all,true_all=confusion_matrix()
save_dir = 'F:\\cut_depth_data\\con_matrix_2c1m_drop0.5_2'
scio.savemat(save_dir, {'predict':predict_all,'true':true_all})
'''''