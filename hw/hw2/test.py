from PIL import Image, ImageChops
import math
import sys
import numpy as np
import hw2
import os
import os.path as osp

if __name__=='__main__':
    option = str(sys.argv[1])
    names = [file.split('.')[0] for file in os.listdir('./') if file.endswith('.txt') and file.startswith('hw2')] if option=='all' else [option]
    allinfo = []
    for name in names:
        hw2.drawfile(name+'.txt')

        test = Image.open(name+' (1).png')
        my = Image.open(name+'.png')
        # diff = ImageChops.difference(test,my)
        # if diff.getbbox():
        #     diff.show()

        width, height = test.size
        image = Image.new('RGBA',(int(width),int(height)),(0,0,0,0))

        info = []

        for x in range(width):
            for y in range(height):
                tp, mp = test.getpixel((x,y)), my.getpixel((x,y))
                if tp == mp:
                    mp = tuple(list(mp)[:3]+[50])
                    image.putpixel((x,y),mp)
                else:
                    if tp != (0,0,0,0) and mp == (0,0,0,0):
                        # 蓝色 画少了
                        image.putpixel((x,y),(0,0,255,255))
                        info.append(', '.join([name,str(x),str(y),'Pixel Missing'])+'\n')
                    elif tp == (0,0,0,0) and mp != (0,0,0,0):
                        # 红色 画多了
                        image.putpixel((x,y),(255,0,0,255))
                        info.append(', '.join([name,str(x),str(y),'Extra Pixel'])+'\n')
                    else:
                        # 颜色不对
                        # tp = tuple(list(tp)[:3]+[50])
                        # image.putpixel((x,y),tp)
                        image.putpixel((x,y),(255,125,50,255))
                        info.append(', '.join([name,str(x),str(y),'Wrong Color'])+'\n')
        # image.save('./COMPRES/'+name+'COMPRESULT.png')
        image.save('./COMPRES/'+name+'COMPRESULT.png')
        allinfo.append(info)

    with open('./COMPRES/result.txt','w') as f:
        for info in allinfo:
            f.writelines(info)
