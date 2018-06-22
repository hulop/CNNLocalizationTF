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
import shutil

def parse_csv(input_file, delimiter=","):
    csv_lines = []
    
    with open(input_file) as fin:
        for line in fin:
            line = line.strip()
            line_tokens = line.split(delimiter)
            csv_lines.append(line_tokens)
    
    return csv_lines

def main():
    parser = argparse.ArgumentParser(description='Places Training')
    parser.add_argument('input_categories_txt', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='Input categories text file.')
    parser.add_argument('input_labels_txt', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='Input image label text file.')
    parser.add_argument('input_image_dir', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='Input image directory.')
    parser.add_argument('output_image_dir', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='Output image directory.')
    args = parser.parse_args()
    input_categories_txt = args.input_categories_txt
    input_labels_txt = args.input_labels_txt
    input_image_dir = args.input_image_dir
    output_image_dir = args.output_image_dir
    print("input categories txt file = " + input_categories_txt)
    print("input labels txt file = " + input_labels_txt)
    print("input image directory = " + input_image_dir)
    print("output image directory = " + output_image_dir)
    
    input_categories = parse_csv(input_categories_txt, delimiter=" ")    
    input_labels = parse_csv(input_labels_txt, delimiter=" ")
    
    categories_id_name_dict = {}
    for input_category in input_categories:
        category_name = os.path.basename(input_category[0])
        category_id = int(input_category[1])
        categories_id_name_dict[category_id] = category_name
    print("input categories id name dict : " + str(categories_id_name_dict))

    for input_label in input_labels:
        input_image_filename = input_label[0]
        input_id = int(input_label[1])
        input_category = categories_id_name_dict[input_id]
        input_image_filepath = os.path.join(input_image_dir, input_image_filename)
        output_image_filepath = os.path.join(output_image_dir, input_category, input_image_filename)                
        print("input image file path : " + input_image_filepath)
        print("input image category : " + input_category)
        print("output image file path : " + output_image_filepath)
        
        if not os.path.exists(os.path.join(output_image_dir, input_category)):
            os.mkdir(os.path.join(output_image_dir, input_category))
        shutil.copyfile(input_image_filepath, output_image_filepath)        
        
if __name__ == '__main__':
    main()
