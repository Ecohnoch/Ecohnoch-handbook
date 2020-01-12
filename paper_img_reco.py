import fitz
import time
import re
import os

arxiv_num = '1804.03641'

download_link = 'https://arxiv.org/pdf/{}.pdf'.format(arxiv_num)



example_pdf_path = '/Users/ecohnoch/Desktop/Data-Processing/References/audio-visual/SoundNet- Learning Sound Representations from Unlabeled Video.pdf'

def pdf2pic(path, pic_path):
    checkXO = r"/Type(?= */XObject)" 
    checkIM = r"/Subtype(?= */Image)"
    doc = fitz.open(path)
    imgcount = 0
    lenXREF = doc._getXrefLength()
    print("文件名:{}, 页数: {}, 对象: {}".format(path, len(doc), lenXREF - 1))

    for i in range(1, lenXREF):
        text = doc._getXrefString(i)
        isXObject = re.search(checkXO, text)
        # 使用正则表达式查看是否是图片
        isImage = re.search(checkIM, text)
        # 如果不是对象也不是图片，则continue
        if not isXObject or not isImage:
            continue
        imgcount += 1
        # 根据索引生成图像
        pix = fitz.Pixmap(doc, i)
        # 根据pdf的路径生成图片的名称
        new_name = path.replace('\\', '_') + "_img{}.png".format(imgcount)
        new_name = new_name.replace(':', '')

        # 如果pix.n<5,可以直接存为PNG
        if pix.n < 5:
            pix.writePNG(os.path.join(pic_path, new_name))
        # 否则先转换CMYK
        else:
            pix0 = fitz.Pixmap(fitz.csRGB, pix)
            pix0.writePNG(os.path.join(pic_path, new_name))
            pix0 = None
        # 释放资源
        pix = None
        print("提取了{}张图片".format(imgcount))


pdf2pic(example_pdf_path, '/Users/ecohnoch/Downloads')