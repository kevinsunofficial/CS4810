from PIL import Image
import math
import sys
import numpy as np


def convertIndex(index):
    """
        convert one-indexed index to zero-indexed format
        -----------------
        ARGUMENTS
        index: one-indexed index
        -----------------
        RETURN
        index: zero-indexed index
    """
    if index > 0:
        index -= 1
    return index


def drawpng(image, lines):
    """
        draw png according to given instruction lines
            read instruction lines one by one, draw or add vertices accordingly
        -----------------
        ARGUMENTS
        image: the image canvas created in drawfile()
        lines: the instruction lines in a list (f.readlines() result)
        -----------------
        RETURN
        image: the image canvas after all instruction lines are performed
    """
    vertices = [] #[[float(x),float(y),int(r),int(g),int(b),int(a)]]
    for line in lines:
        instr = line.strip().split()
        if instr:
            kw = instr[0]

            if kw in ['xy','xyrgb','xyc','xyrgba']:
                if kw=='xy':
                    x, y = [float(i) for i in instr[1:]]
                    r, g, b, a = 255, 255, 255, 255
                elif kw=='xyrgb':
                    x, y = [float(i) for i in instr[1:3]]
                    r, g, b = [int(i) for i in instr[3:]]
                    a = 255
                elif kw=='xyrgba':
                    x, y = [float(i) for i in instr[1:3]]
                    r, g, b, a = [int(i) for i in instr[3:]]
                elif kw =='xyc':
                    x, y = [float(i) for i in instr[1:3]]
                    hexColor = instr[3]
                    r, g, b = [int('0x'+hexColor[i:i+2],16) for i in range(1,len(hexColor),2)]
                    a = 255
                vertices.append([x, y, r, g, b, a])
            
            elif kw in ['linec','lineca','lineg','tric','trica','trig','cubicc','cubicg']:
                if kw == 'linec':
                    p1i, p2i = [int(i) for i in instr[1:3]]
                    hexColor = instr[3]
                    r, g, b = [int('0x'+hexColor[i:i+2],16) for i in range(1,len(hexColor),2)]
                    a = 255
                    p1, p2 = vertices[convertIndex(p1i)], vertices[convertIndex(p2i)]
                    p1[2:], p2[2:] = [r, g, b, a], [r, g, b, a]
                    image = drawline(image,p1,p2)

                elif kw == 'lineca':
                    p1i, p2i = [int(i) for i in instr[1:3]]
                    hexColor = instr[3]
                    r, g, b, a = [int('0x'+hexColor[i:i+2],16) for i in range(1,len(hexColor),2)]
                    p1, p2 = vertices[convertIndex(p1i)], vertices[convertIndex(p2i)]
                    p1[2:], p2[2:] = [r, g, b, a], [r, g, b, a]
                    image = drawline(image,p1,p2)

                elif kw == 'lineg':
                    p1i, p2i = [int(i) for i in instr[1:3]]
                    p1, p2 = vertices[convertIndex(p1i)], vertices[convertIndex(p2i)]
                    image = drawline(image,p1,p2)


                elif kw == 'tric':
                    p1i, p2i, p3i = [int(i) for i in instr[1:4]]
                    hexColor = instr[4]
                    r, g, b = [int('0x'+hexColor[i:i+2],16) for i in range(1,len(hexColor),2)]
                    p1, p2, p3 = vertices[convertIndex(p1i)],vertices[convertIndex(p2i)],vertices[convertIndex(p3i)]
                    a = 255
                    p1[2:], p2[2:], p3[2:] = [r, g, b, a], [r, g, b, a], [r, g, b, a]
                    image = drawtrig(image,p1,p2,p3)

                elif kw == 'trica':
                    p1i, p2i, p3i = [int(i) for i in instr[1:4]]
                    hexColor = instr[4]
                    r, g, b, a = [int('0x'+hexColor[i:i+2],16) for i in range(1,len(hexColor),2)]
                    p1, p2, p3 = vertices[convertIndex(p1i)],vertices[convertIndex(p2i)],vertices[convertIndex(p3i)]
                    p1[2:], p2[2:], p3[2:] = [r, g, b, a], [r, g, b, a], [r, g, b, a]
                    image = drawtrig(image,p1,p2,p3)

                elif kw == 'trig':
                    p1i, p2i, p3i = [int(i) for i in instr[1:4]]
                    p1, p2, p3 = vertices[convertIndex(p1i)],vertices[convertIndex(p2i)],vertices[convertIndex(p3i)]
                    image = drawtrig(image,p1,p2,p3)

                elif kw == 'cubicc':
                    psi = [int(i) for i in instr[1:-1]]
                    hexColor = instr[-1]
                    r, g, b = [int('0x'+hexColor[i:i+2],16) for i in range(1,len(hexColor),2)]
                    a = 255
                    ps = [vertices[convertIndex(i)] for i in psi]
                    for p in ps:
                        p[2:] = [r, g, b, a]
                    image = drawcubic(image, ps)

                elif kw == 'cubicg':
                    psi = [int(i) for i in instr[1:]]
                    ps = [vertices[convertIndex(i)] for i in psi]
                    image = drawcubic(image, ps)

            elif kw in ['fann', 'stripn']:
                psi = [int(i) for i in instr[2:]]
                ps = [vertices[convertIndex(i)] for i in psi]

                if kw == 'fann':                    
                    image = drawfann(image,ps)

                elif kw == 'stripn':
                    image = drawstripn(image,ps)
    return image


def drawline(image, p1, p2):
    """
        draw 8-connected line with DDA algorithm
        -----------------
        ARGUMENTS
        image: the image canvas
        p1: the first vertex
        p2: the second vertex
        hexColor: the color code (r, g, b, a)
        -----------------
        RETURN
        image: the image canvas after the line is drawn
    """
    p1, p2 = np.array(p1), np.array(p2)
    dp = p2-p1 # [dx, dy]
    use = 0 if abs(dp[0])>=abs(dp[1]) else 1
    d = abs(dp[use])
    if d:
        step = dp/d
        if p2[use]<p1[use]:
            p1, p2 = p2, p1
            step = -step
        s = (math.ceil(p1[use])-p1[use])/dp[use]
        q = p1+s*dp
        while q[use]<p2[use]:
            dot = q[:2]
            hexColor = tuple(q[2:].astype(int))
            coor = (int(math.floor(dot[~use]+0.5)),int(dot[use])) if use else (int(dot[use]),int(math.floor(dot[~use]+0.5)))
            px = image.im.getpixel(coor)
            if px != (0,0,0,0):
                Cb, ab, Ca, aa = np.array(list(px)[:-1]), list(px)[-1]/255, np.array(list(hexColor)[:-1]), list(hexColor)[-1]/255
                ao = aa+ab*(1-aa)
                Co = (Ca*aa+Cb*ab*(1-aa))/ao
                hexColor = tuple([int(f) for f in Co.tolist()+[ao*255]])
            image.im.putpixel(coor,hexColor)
            q += step
    
    return image


def ydda(p1, p2):
    """
        Helper function with DDA algorithm steping in y
        -----------------
        ARGUMENTS
        p1: the first vertex
        p2: the second vertex
        -----------------
        RETURN
        qs: all dots along the DDA drawn line
    """
    qs = []
    p1, p2 = np.array(p1), np.array(p2)
    dp = p2-p1
    d = abs(dp[1])
    if d:
        step = dp/d
        if p2[1]<p1[1]:
            p1, p2 = p2, p1
            step = -step
        s = (math.ceil(p1[1])-p1[1])/dp[1]
        q = p1+s*dp
        while q[1]<p2[1]:
            qs.append(tuple(q.tolist()))
            q += step
    return qs


def drawtrig(image, p1, p2, p3):
    """
        draw and fill triangle with DDA algorithm
        -----------------
        ARGUMENTS
        image: the image canvas
        p1: the first vertex
        p2: the second vertex
        p3: the third vertex
        -----------------
        RETURN
        image: the image canvas after the triangle is drawn
    """
    p1, p2, p3 = sorted([p1,p2,p3],key=lambda p: p[1])
    p12, p13, p23 = ydda(p1,p2), ydda(p1,p3), ydda(p2,p3)
    allp = sorted(p12+p13+p23,key=lambda q: q[1])
    for i in range(0,len(allp),2):
        q1, q2 = allp[i], allp[i+1]
        image = drawline(image, q1, q2)
    
    return image


def drawfann(image, ps):
    """
        Goaraud fill a polygon with adjacent pair of subsequent vertices
        -----------------
        ARGUMENTS
        image: the image canvas
        ps: all vertices
        -----------------
        RETURN
        image: the image canvas after the polygon is drawn
    """
    p1, pf = ps[0], ps[1:]
    for i in range(len(pf)-1):
        p2, p3 = pf[i], pf[i+1]
        image = drawtrig(image, p1, p2, p3)
    return image


def drawstripn(image, ps):
    """
        Goaraud fill a polygon with three consecutive indices
        -----------------
        ARGUMENTS
        image: the image canvas
        ps: all vertices
        -----------------
        RETURN
        image: the image canvas after the polygon is drawn
    """
    for i in range(len(ps)-2):
        p1, p2, p3 = ps[i], ps[i+1], ps[i+2]
        image = drawtrig(image, p1, p2, p3)
    return image


def onestepbezier(ps):
    """
        Find seven new points of given four vertices for bezier curve
        -----------------
        ARGUMENTS
        ps: all vertices, containing four points
        -----------------
        RETURN
        qs1: first set of new four vertices
        qs2: second set of new four vertices
    """
    qs = [np.array(p) for p in ps]
    newqs = [qs[0], qs[-1]]
    for k in range(1,len(qs)):
        temp = []
        for i in range(len(qs)-k):
            temp.append(0.5*qs[i]+0.5*qs[i+1])
            qs[i] = 0.5*qs[i]+0.5*qs[i+1]
        newqs.extend([temp[0], temp[-1]])
    mid = qs[0]
    qs1, qs2 = [], []
    for i in range(0,len(newqs[:-1]),2):
        qs1.append(newqs[i])
        qs2.append(newqs[i+1])
    qs2.reverse()

    return qs1, qs2


def distance(ps):
    """
        Calculate max distance of the given four bezier control points
            Assume p1 and p4 are starting and ending point
        -----------------
        ARGUMENTS
        ps: all vertices, containing four points
        -----------------
        RETURN
        val: the max distance of the given points
    """
    qs = [np.array(p[:2]) for p in ps]
    p1, p2, p3, p4 = qs
    a1, a2 = p4-p1, p1-p4
    b1, b2 = p2-p1, p3-p4
    if np.array_equal(p1,p4) and (not np.array_equal(p2,p3)):
        return np.inf
    abnorm1, abnorm2 = np.linalg.norm(a1)*np.linalg.norm(b1), np.linalg.norm(a2)*np.linalg.norm(b2)
    abcross1, abcross2 = np.cross(a1,b1), np.cross(a2,b2)
    return max([abcross1/abnorm1, abcross2/abnorm2])


def drawcubic(image, ps, threshold=0.01):
    """
        Draw cubic bezier curve using four control points
        -----------------
        ARGUMENTS
        image: the image canvas
        ps: all vertices, containing four points
        -----------------
        RETURN
        image: the image canvas after the cubic bezier curve is drawn
    """
    ps = [np.array(p) for p in ps]
    d = distance(ps)
    if d < threshold:
        image = drawline(image, ps[0], ps[-1])
    else:
        qs1, qs2 = onestepbezier(ps)
        image = drawcubic(image, qs1)
        image = drawcubic(image, qs2)

    return image


def drawfile(file):
    """
        draw the contents specified in one input file
            read first line of the file and pass the rest of the instructions
            to drawpng()
        -----------------
        ARGUMENTS
        file: file name of the input file
    """
    with open(file,'r') as f:
        frameinstr = f.readline().strip().split()
        if frameinstr[0]=='png':
            kw, width, height, filename = frameinstr
            image = Image.new('RGBA',(int(width),int(height)),(0,0,0,0))
            image = drawpng(image,f.readlines())
            image.save(filename)


if __name__ == "__main__":
    drawfile(sys.argv[1])