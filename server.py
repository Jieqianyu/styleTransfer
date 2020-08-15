from PIL import Image
from flask import Flask, request, render_template
from glob import glob
import os
from imageTransfer import Transfer
import numpy as np

class PathInfo(object):
    def __init__(self):
        style_img_paths = glob('static/style/*.jpg') + glob('static/style/*.jpeg') + glob('static/style/*.png')
        self.styles = [(path, os.path.basename(path)) for path in style_img_paths]
        self.content_img_path = None
        self.style_img_path = None
        self.transfer_img_path =None
        
path_info = PathInfo()
transfer = Transfer()

def process(path_info):
    transfer.load_data(path_info.style_img_path, path_info.content_img_path)
    res = transfer.transfer()
    print(res.shape)
    return Image.fromarray(res.astype('uint8')).convert('RGB')


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Request for content image
        try:
            content_file = request.files['content']
        except Exception as e:
            print(e)
        else:
            # Save content image
            content_img = Image.open(content_file.stream)  # PIL image
            path_info.content_img_path = "static/content/" + content_file.filename
            # print(path_info.content_img_path)
            content_img.save(path_info.content_img_path)

        return render_template('index.html',
                                styles=path_info.styles,
                                content=path_info.content_img_path,
                                transfer=path_info.transfer_img_path)
    else:
        # Request for style image
        try:
            style_basename = request.values['style_basename']
            print(style_basename)
        except Exception as e:
            print(e)
        else:
            path_info.style_img_path = 'static/style/%s' % style_basename
            # process images
            transfer = process(path_info)
            # Save transfer image
            path_info.transfer_img_path = "static/transfer/%s_stylized_%s.png" % (
                os.path.splitext(os.path.basename(path_info.content_img_path))[0],
                os.path.splitext(os.path.basename(path_info.style_img_path))[0])
            print(path_info.transfer_img_path)
            transfer.save(path_info.transfer_img_path)
            
        return render_template('index.html',
                                styles=path_info.styles,
                                content=path_info.content_img_path,
                                transfer=path_info.transfer_img_path)

if __name__=="__main__":
    app.run("127.0.0.1", 7777)
