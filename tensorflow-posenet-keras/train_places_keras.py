##############################################################################
#The MIT License (MIT)
#
#Copyright (c) 2018 IBM Corporation, Carnegie Mellon University and others
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
##############################################################################

import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

### multi GPU setting ###
#batch_size = 256
#num_gpus = 4
### single GPU setting ###
batch_size = 64
num_gpus = 1
##########################
epochs = 90
use_augmentation = True
img_width, img_height = 299, 299

def parse_csv(input_file, delimiter=","):
    csv_lines = []
    
    with open(input_file) as fin:
        for line in fin:
            line = line.strip()
            line_tokens = line.split(delimiter)
            csv_lines.append(line_tokens)
    
    return csv_lines

def read_category_names(input_categories_txt):
    category_names = []
    input_categories = parse_csv(input_categories_txt, delimiter=" ")
    for input_category in input_categories:
        category_name = os.path.basename(input_category[0])
        category_names.append(category_name)
    return category_names
    
def main():
    parser = argparse.ArgumentParser(description='Places Training')
    parser.add_argument('input_categories_txt', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='Input categories text file.')
    parser.add_argument('train_data_dir', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='Training directory path of Places dataset.')
    parser.add_argument('val_data_dir', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='Validation directory path of Places dataset.')
    parser.add_argument('output_model_dir', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='Output model directory.')
    parser.add_argument('output_log_dir', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='Output log directory.')
    args = parser.parse_args()
    input_categories_txt = args.input_categories_txt
    train_data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir    
    output_model_dir = args.output_model_dir
    output_log_dir = args.output_log_dir
    print("input categories txt file = " + input_categories_txt)
    print("train data directory = " + train_data_dir)
    print("validation data directory = " + val_data_dir)
    print("output model directory = " + output_model_dir)
    print("output log directory = " + output_log_dir)
    
    category_names = read_category_names(input_categories_txt)
    print("category names = " + str(category_names))
    
    # prepare data augmentation configuration
    if use_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        classes=category_names,
        batch_size=batch_size*num_gpus,
        class_mode='categorical')
    
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        classes=category_names,
        batch_size=batch_size*num_gpus,
        class_mode='categorical')
    
    print("train samples=" + str(train_generator.samples))
    print("train number of classes=" + str(train_generator.num_classes))
    print("validation samples=" + str(val_generator.samples))
    print("validation classes=" + str(val_generator.num_classes))
    
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # add top layer
    model_output = base_model.output
    model_output = GlobalAveragePooling2D(name='avg_pool')(model_output)
    model_predictions = Dense(train_generator.num_classes, activation='softmax', name='predictions')(model_output)
    
    model = Model(input=base_model.input, output=model_predictions)
    if num_gpus>1:
        model = multi_gpu_model(model, gpus=num_gpus)
    
    #Save the model after every epoch.
    checkpointer = ModelCheckpoint(os.path.join(output_model_dir, "checkpoint_weights.h5"), verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    
    #Save the TensorBoard logs.
    logger = TensorBoard(log_dir=output_log_dir, histogram_freq=0, write_graph=True, write_images=True)
    
    #Compile model
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=0.00000001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    #Train model
    train_steps_per_epoch = int(train_generator.samples/float(batch_size*num_gpus))
    validation_steps = int(val_generator.samples/float(batch_size*num_gpus))
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        workers=8,
        callbacks=[checkpointer, logger])
    
    model.save_weights(os.path.join(output_model_dir, "trained_weights.h5"))
    
if __name__ == '__main__':
    main()
