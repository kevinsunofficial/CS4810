from PIL import Image
import sys


def drawpng(image, line):
    instr = line.strip().split()
    kw = instr[0]
    if kw=='xy':
        x, y = [int(i) for i in instr[1:]]
        r, g, b = 255, 255, 255
    elif kw=='xyrgb':
        x, y, r, g, b = [int(i) for i in instr[1:]]
    elif kw =='xyc':
        x, y = [int(i) for i in instr[1:3]]
        hexColor = instr[3]
        r, g, b = [int('0x'+hexColor[i:i+2],16) for i in range(1,len(hexColor),2)]
    image.im.putpixel((x,y),(r,g,b,255))
    return image


def drawpngs(images, file):
    instrs = [i.strip().split() for i in file.strip().split('frame') if i]
    for instr in instrs:
        k, line = instr[0], '\t'.join(instr[1:])
        images[k] = drawpng(images[k],line)
    return images


def drawfile(file):
    with open(file,'r') as f:
        frameinstr = f.readline().strip().split()
        if frameinstr[0]=='png':
            kw, width, height, filename = frameinstr
            image = Image.new('RGBA',(int(width),int(height)),(0,0,0,0))
            for line in f.readlines():
                image = drawpng(image,line)
            image.save(filename)
        else:
            kw, width, height, basename, numframe = frameinstr
            images = {
                str(i): Image.new('RGBA',(int(width),int(height)),(0,0,0,0)) for i in range(int(numframe))
            }
            images = drawpngs(images, f.read())
            for k in images.keys():
                images[k].save(basename+'%03d'%int(k)+'.png')


if __name__ == "__main__":
    drawfile(sys.argv[1])
