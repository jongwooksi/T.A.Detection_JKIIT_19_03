import tensorflow as tf

import unet
import pix2pix

from flownet2.src.flowlib import flow_to_image
from flownet2.src.flownet_sd.flownet_sd import FlowNetSD  # Ok
from flownet2.src.training_schedules import LONG_SCHEDULE
from flownet2.src.net import Mode

from monodepth.monodepth_model import *

slim = tf.contrib.slim
#slim = tf.train


def generator(inputs, layers, features_root=64, filter_size=3, pool_size=2, output_channel=3):
    return unet.unet(inputs, layers, features_root, filter_size, pool_size, output_channel)


def discriminator(inputs, num_filers=(128, 256, 512, 512)):
    logits, end_points = pix2pix.pix2pix_discriminator(inputs, num_filers)
    return logits, end_points['predictions']


def flownet(input_a, input_b, height, width, reuse=None):
    net = FlowNetSD(mode=Mode.TEST)
    # train preds flow
    input_a = (input_a + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    input_b = (input_b + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    
    #input_a = tf.image.central_crop(input_a, 0.005)
    #input_b = tf.image.central_crop(input_b, 0.005)
    input_a = tf.image.crop_to_bounding_box(input_a, 124, 58, 4, 12)
    input_b = tf.image.crop_to_bounding_box(input_b, 124, 58, 4, 12)
    print("aaaaa", input_a)
    print("aaaaa", input_b)

    # input size is 384 x 512
    input_a = tf.image.resize_images(input_a, [height, width])
    input_b = tf.image.resize_images(input_b, [height, width])
    flows = net.model(
        inputs={'input_a': input_a, 'input_b': input_b},
        training_schedule=LONG_SCHEDULE,
        trainable=False, reuse=reuse
    )
    print(flows['flow'].shape)
    return flows['flow']


def initialize_flownet(sess, checkpoint):
    flownet_vars = slim.get_variables_to_restore(include=['FlowNetSD'])
    
    
    flownet_saver = tf.train.Saver(flownet_vars)
    #for i, var in enumerate(flownet_saver._var_list):
    #    print('Var {}: {}'.format(i, var))
    print('FlownetSD restore from {}!'.format(checkpoint))
    flownet_saver.restore(sess, checkpoint)


def monodepth(input_image, height, width):
    
    params = monodepth_parameters(
        encoder="vgg",
        height=height,
        width=width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    print(input_image.shape)
    #input_image = input_image  / 255.0
    #input_image = np.stack((input_image, np.fliplr(input_image)), 0)
    #left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3]) 
    
    
    input_image = tf.image.resize_images(input_image, [height, width])
    #input_image = tf.dtypes.cast(input_image, tf.float32)
    #input_image = input_image.astype(np.float32) / 255
    input_image =  tf.image.convert_image_dtype(input_image, dtype = tf.float32)
    input_image2 = tf.image.flip_left_right(input_image)

    print(input_image.shape)
    print(input_image2.shape)

    input_images = tf.concat([input_image, input_image2], 0)
    print(input_images.shape)
    #left  = tf.placeholder(tf.float32, [2,height, width, 3])
    #print(input_image.shape)
    net = MonodepthModel(params, "test", input_images, True)
    #for v in tf.global_variables():
    #        print(v)
    #print(net)
    disp = net.build_outputs(input_images)
    #print(disp)
    #mono = net.build_model()
    
    return disp


def initialize_monodepth(sess, checkpoint):
    #sess.run(tf.local_variables_initializer())
    #monodepth_vars = slim.get_variables_to_restore(include=['MonodepthModel'])
    #monodepth_saver=tf.train.import_meta_graph(checkpoint+".meta")
    monodepth_vars = [v for v in tf.global_variables() if (v.name).split('/')[0] == "model"]
    #print(var_23)
    monodepth_saver = tf.train.Saver(monodepth_vars)


    #tf.reset_default_graph()
    for i, var in enumerate(monodepth_saver._var_list):
        print('Var {}: {}'.format(i, var))

    coordinator = tf.train.Coordinator()
    sess.run(tf.local_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
    print('monodepth restore from {}!'.format(checkpoint))
    monodepth_saver.restore(sess, checkpoint)
    #disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
   
