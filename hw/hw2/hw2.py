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


def drawpng(image, dim, lines):
    """
        draw png according to given instruction lines
            read instruction lines one by one, draw or add vertices accordingly
        -----------------
        ARGUMENTS
        image: the image canvas created in drawfile()
        dim: dimension of the image canvas, with (width, height)
        lines: the instruction lines in a list (f.readlines() result)
        -----------------
        RETURN
        image: the image canvas after all instruction lines are performed
    """
    width, height = dim
    raster = np.zeros((8, height, width)) # x, y, z, w, r, g, b, a
    zbuff = np.ones(dim)
    vertices = [] #[[float(x),float(y),float(z),float(w)]]
    color = {
        'r':255, 'g':255, 'b':255, 'a':255
    }
    mv, proj = np.matrix([]), np.matrix([])
    cull = False
    for line in lines:
        instr = line.strip().split()
        if instr:
            kw = instr[0]
            if kw in ['xyz', 'xyzw']:
                if kw=='xyz':
                    x, y, z = [float(i) for i in instr[1:]]
                    w = 1
                    r, g, b, a = [color['r'], color['g'], color['b'], color['a']]
                elif kw=='xyzw':
                    x, y, z, w = [float(i) for i in instr[1:]]
                    r, g, b, a = [color['r'], color['g'], color['b'], color['a']]
                vertices.append([x, y, z, w, r, g, b, a])
            elif kw in ['trif', 'trig']:
                p1i, p2i, p3i = [int(i) for i in instr[1:4]]
                p1, p2, p3 = np.array(vertices[convertIndex(p1i)]),np.array(vertices[convertIndex(p2i)]),np.array(vertices[convertIndex(p3i)])
                if kw=='trif':
                    rgba = [color['r'], color['g'], color['b'], color['a']]
                    p1[4:], p2[4:], p3[4:] = rgba, rgba, rgba
                raster, zbuff = operateVtx(p1, p2, p3, mv, proj, dim, raster, zbuff, cull)
            elif kw=='color':
                temp = [float(i) for i in instr[1:]]
                # print('before',temp)
                for c in range(len(temp)):
                    if 0 <= temp[c] <= 1:
                        temp[c] = int(temp[c] * 255)
                    else:
                        temp[c] = 0 if temp[c] < 0 else 255
                # print('after',temp)
                color['r'], color['g'], color['b'] = temp
            elif kw=='loadmv':
                mv = np.array([float(i) for i in instr[1:]]).reshape(4,4)
            elif kw=='loadp':
                proj = np.array([float(i) for i in instr[1:]]).reshape(4,4)

            elif kw in ['frustum','ortho']:
                l, r, b, t, n, f = [float(i) for i in instr[1:]]
                if kw=='frustum':
                    proj = np.matrix([
                        [2*n/(r-l),0,(r+l)/(r-l),0],
                        [0,2*n/(t-b),(t+b)/(t-b),0],
                        [0,0,-(f+n)/(f-n),-2*f*n/(f-n)],
                        [0,0,-1,0]
                    ])
                elif kw=='ortho':
                    n = 2*n-f
                    proj = np.matrix([
                        [2/(r-l),0,0,-(r+l)/(r-l)],
                        [0,2/(t-b),0,-(t+b)/(t-b)],
                        [0,0,-2/(f-n),-(f+n)/(f-n)],
                        [0,0,0,1]
                    ])
            
            elif kw in ['translate','rotatex','rotatey','rotatez','rotate','scale']:
                if kw=='translate':
                    dx, dy, dz = [float(i) for i in instr[1:]]
                    t = np.matrix([
                        [1,0,0,dx],
                        [0,1,0,dy],
                        [0,0,1,dz],
                        [0,0,0,1]
                    ])
                    mv = np.matmul(mv,t) if mv.size else t
                elif kw=='rotate':
                    dgr, x, y, z = [float(i) for i in instr[1:]]
                    dgr = dgr/180*np.pi
                    v = np.array([x,y,z])
                    x, y, z = v/np.linalg.norm(v)
                    sin, cos = np.sin(dgr), np.cos(dgr)
                    r = np.matrix([
                        [x**2*(1-cos)+cos,  x*y*(1-cos)-z*sin,  x*z*(1-cos)+y*sin,  0],
                        [x*y*(1-cos)+z*sin, y**2*(1-cos)+cos,   y*z*(1-cos)-x*sin,  0],
                        [x*z*(1-cos)-y*sin, y*z*(1-cos)+x*sin,  z**2*(1-cos)+cos,   0],
                        [0,                 0,                  0,                  1]
                    ])
                    mv = np.matmul(mv,r) if mv.size else r
                elif kw in ['rotatex', 'rotatey', 'rotatez']:
                    axis, dgr = kw[-1], float(instr[-1])/180*np.pi
                    if axis=='x':
                        r = np.matrix([
                            [1,0,0,0],
                            [0,np.cos(dgr),-np.sin(dgr),0],
                            [0,np.sin(dgr),np.cos(dgr),0],
                            [0,0,0,1]
                        ])
                    elif axis=='y':
                        r = np.matrix([
                            [np.cos(dgr),0,np.sin(dgr),0],
                            [0,1,0,0],
                            [-np.sin(dgr),0,np.cos(dgr),0],
                            [0,0,0,1]
                        ])
                    else:
                        r = np.matrix([
                            [np.cos(dgr),-np.sin(dgr),0,0],
                            [np.sin(dgr),np.cos(dgr),0,0],
                            [0,0,1,0],
                            [0,0,0,1]
                        ])
                    mv = np.matmul(mv,r) if mv.size else r
                elif kw=='scale':
                    sx, sy, sz = [float(i) for i in instr[1:]]
                    s = np.matrix([
                        [sx,0,0,0],
                        [0,sy,0,0],
                        [0,0,sz,0],
                        [0,0,0,1]
                    ])
                    mv = np.matmul(mv,s) if mv.size else s
            
            elif kw=='lookat':
                ei, ci = [int(i) for i in instr[1:3]]
                e, c = np.array(vertices[convertIndex(ei)][:3]),np.array(vertices[convertIndex(ci)][:3])
                up = np.array([float(i) for i in instr[3:]])
                f = c-e
                f, up = f/np.linalg.norm(f), up/np.linalg.norm(up)
                s = np.cross(f,up)
                s = s/np.linalg.norm(s)
                u = np.cross(s,f)
                l = np.matrix([
                    [s[0],s[1],s[2],0],
                    [u[0],u[1],u[2],0],
                    [-f[0],-f[1],-f[2],0],
                    [0,0,0,1]
                ])
                t = np.matrix([
                    [1,0,0,-e[0]],
                    [0,1,0,-e[1]],
                    [0,0,1,-e[2]],
                    [0,0,0,1]
                ])
                mv = np.matmul(l,t)
            
            elif kw=='multmv':
                nm = np.matrix([float(i) for i in instr[1:]]).reshape(4,4)
                mv = np.matmul(mv,nm) if mv.size else nm
            
            elif kw=='cull':
                cull = True

    image = drawRaster(image, raster)
    return image


def operateVtx(p1, p2, p3, mv, proj, dim, raster, zbuff, cull):
    width, height = dim
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    if mv.size:
        p1[:4], p2[:4], p3[:4] = np.matmul(mv,p1[:4]), np.matmul(mv,p2[:4]), np.matmul(mv,p3[:4])
    if proj.size:
        p1[:4], p2[:4], p3[:4] = np.matmul(proj,p1[:4]), np.matmul(proj,p2[:4]), np.matmul(proj,p3[:4])
    p1[:3], p2[:3], p3[:3] = p1[:3]/p1[3], p2[:3]/p2[3], p3[:3]/p3[3]
    p1[0], p2[0], p3[0] = (p1[0]+1)*width/2, (p2[0]+1)*width/2, (p3[0]+1)*width/2
    p1[1], p2[1], p3[1] = (p1[1]+1)*height/2, (p2[1]+1)*height/2, (p3[1]+1)*height/2

    if cull:
        dp21, dp32 = p2-p1, p3-p2
        if dp32[1]/dp32[0] > dp21[1]/dp21[0]:
            raster, zbuff = rstrTrig(dim, p1, p2, p3, raster, zbuff)
    else:
        raster, zbuff = rstrTrig(dim, p1, p2, p3, raster, zbuff)
    return raster, zbuff


def rstrTrig(dim, p1, p2, p3, raster, zbuff):
    width, height = dim
    p1, p2, p3 = sorted([p1,p2,p3],key=lambda p: p[1])
    p12, p13, p23 = ydda(p1,p2), ydda(p1,p3), ydda(p2,p3)
    allp = sorted(p12+p13+p23,key=lambda q: q[1])
    for i in range(0,len(allp),2):
        q1, q2 = allp[i], allp[i+1]
        raster, zbuff = itrpLine(dim, q1, q2, raster, zbuff)
    return raster, zbuff


def itrpLine(dim, p1, p2, raster, zbuff):
    width, height = dim
    p1, p2 = np.array(p1), np.array(p2)
    dp = p2-p1 # [dx, dy]
    use = 0 if abs(dp[0])>=abs(dp[1]) else 1
    d = abs(dp[use])
    if d >= 1E-14:
        if dp[use]<0:
            p1, p2 = p2, p1
            dp = -dp
        step = dp/d
        s = (math.ceil(p1[use])-p1[use])/dp[use]
        q = p1+s*dp
        while p2[use]-q[use]>0:
            dot = q[:2]
            dotz, dotw = q[2:4]
            coor = (int(math.floor(dot[~use]+0.5)),int(dot[use])) if use else (int(dot[use]),int(math.floor(dot[~use]+0.5)))
            # print(coor)
            if 0<=coor[1]<height and 0<=coor[0]<width:
                if 0 <= dotz <= 1 and dotz <= zbuff[coor]:
                    raster[:,coor[1],coor[0]] = [coor[0],coor[1],dotz,dotw]+q[4:].astype(int).tolist()
                    zbuff[coor] = dotz
            q += step
    
    return raster, zbuff


def ydda(p1, p2):
    p1, p2 = np.array(p1), np.array(p2)
    qs = []
    dp = p2-p1
    d = abs(dp[1])
    if d >= 1E-14:
        if dp[1]<0:
            p1, p2 = p2, p1
            dp = -dp
        step = dp/d
        s = (math.ceil(p1[1])-p1[1])/dp[1]
        q = p1+s*dp
        while p2[1]-q[1]>0:
            qs.append(q.copy())
            q += step
    return qs


def drawRaster(image, raster):
    R, Y, X = raster.shape
    for x in range(X):
        for y in range(Y):
            coor = tuple(raster[:2,y,x].astype(int).tolist())
            hexColor = tuple(raster[4:,y,x].astype(int).tolist())
            if hexColor != (0,0,0,0):
                # if coor[0]==13 and 79<=coor[1]<=79: hexColor=(255,0,0,255)
                image.im.putpixel(coor,hexColor)
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
            image = drawpng(image,(int(width),int(height)),f.readlines())
            image.save(filename)


if __name__ == "__main__":
    drawfile(sys.argv[1])
