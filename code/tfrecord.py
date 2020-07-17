import os,sys
import tensorflow as tf
import gdal

data_dir_train='F:/Anais/NL_davy/images_train_NL'
#data_dir_test='F:/Anais/PSGAN/data/images_test'
#testlist=['%s/%d'%(data_dir_test,number) for number in range(3155,3537)]
trainfiles=['%s/%d'%(data_dir_train,number) for number in range(3154)]
output_dir="F://Anais/NL_davy/tf_records"

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(inputfiles, name):
    num_examples=len(inputfiles)
    print(num_examples)
    filename=os.path.join(output_dir,name+'.tfrecords')
    print ('Writing', filename)
    writer=tf.python_io.TFRecordWriter(filename)
    for (file,i) in zip(inputfiles, range(num_examples)):
        
        if i%100 == 0:
            print (file,i)
  
        img_name = '%s_%d' % (name, i)
        mul_filename = '%s_mul.tif' % file
        blur1_filename = '%s_lr_u.tif' % file
        pan_filename = '%s_pan.tif' % file

        im_mul_raw = gdal.Open(mul_filename).ReadAsArray().transpose(1, 2, 0).tostring()
        im_blur_raw = gdal.Open(blur1_filename).ReadAsArray().transpose(1, 2, 0).tostring()
        im_pan_raw = gdal.Open(pan_filename).ReadAsArray().reshape([128, 128, 1]).tostring()
        
        example = tf.train.Example(features=tf.train.Features(feature={
            #'im_name': _bytes_feature(img_name),
            'im_mul_raw': _bytes_feature(im_mul_raw),
            'im_blur_raw':_bytes_feature(im_blur1_raw),
            'im_pan_raw':_bytes_feature(im_pan_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

    
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
convert_to(trainfiles,'train')
#convert_to(testlist,'test')



