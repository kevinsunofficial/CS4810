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


def convertColor(color, v):
    sRGBcolor = []
    for l in color:
        l = 1-np.exp(-l*v) if v else l
        ls = 12.92*l if l <= 0.0031308 else 1.055*(l**(1/2.4))-0.055
        sRGBcolor.append(int(ls*255))
    return tuple(sRGBcolor)


def objEqual(obj1, obj2):
    if obj1[0] == obj2[0]:
        if obj1[0] == 'sphere':
            return np.all(obj1[1])==np.all(obj2[1]) and obj1[2]==obj2[2]
        elif obj1[0] == 'plane':
            return np.all(np.array([obj1[1:5]]))==np.all(np.array(obj2[1:5]))
    return False



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
    objs = []
    obj_color = [1,1,1] # [r,g,b]
    suns = [] # [[x,y,z,r,g,b]]
    bulbs = [] # [[x,y,z,r,g,b]]
    v = 0

    for line in lines:
        instr = line.strip().split()
        if instr:
            kw = instr[0]
            temp = [float(i) for i in instr[1:]]
            if kw in ['sphere','plane']:
                o = [kw]+temp+obj_color
                objs.append(o)
            elif kw=='sun':
                suns.append(temp+obj_color)
            elif kw=='bulb':
                bulbs.append(temp+obj_color)
            elif kw=='color':
                obj_color = temp
            elif kw=='expose':
                v = float(instr[1])
    image = drawRaster(image, dim, objs, suns, bulbs, v)
    return image


def raySphere(ro, rd, c, r):
    inside = np.linalg.norm(c-ro)**2 < r**2
    tc = np.dot((c-ro),rd)/np.linalg.norm(rd)
    if not inside and tc < 0:
        return np.inf
    d2 = np.linalg.norm(ro+tc*rd-c)**2
    if not inside and d2 > r**2:
        return np.inf
    toff = np.sqrt(r**2-d2)/np.linalg.norm(rd)
    t = (tc+toff) if inside else (tc-toff)
    return t


def sunContactSphere(ro, rd, c, r):
    inside = np.linalg.norm(c-ro)**2 < r**2
    tc = np.dot((c-ro),rd)/np.linalg.norm(rd)
    if not inside and tc < 0:
        return False
    d2 = np.linalg.norm(ro+tc*rd-c)**2
    if not inside and d2 > r**2:
        return False
    return True


def bulbContactSphere(ro, rd, c, r, bulbdist):
    inside = np.linalg.norm(c-ro)**2 < r**2
    tc = np.dot((c-ro),rd)/np.linalg.norm(rd)
    if not inside and tc < 0:
        return False
    d2 = np.linalg.norm(ro+tc*rd-c)**2
    if not inside and d2 > r**2:
        return False
    toff = np.sqrt(r**2-d2)/np.linalg.norm(rd)
    t = (tc+toff) if inside else (tc-toff)
    return t>0 and np.linalg.norm(t*rd)<bulbdist


def invSphere(ro, rd, t, c, r, obj_color, suns, bulbs, obstacles):
    # print(ro,rd,t,c,r,obj_color)
    color = np.array([0,0,0])
    p = ro+t*rd
    normal = (p-c)/r
    normal = -normal if np.dot(normal,rd)>0 else normal

    if len(suns):
        for sun in suns:
            sun_dir, sun_color = np.array(sun[:3]), np.array(sun[3:])
            sun_dir = sun_dir/np.linalg.norm(sun_dir)
            
            dark = False
            if obstacles:
                for obj in obstacles:
                    if obj[0] == 'sphere':
                        dark = sunContactSphere(p, sun_dir, obj[1], obj[2])
                    if obj[0] == 'plane':
                        dark = sunContactPlane(p, sun_dir, obj[1], obj[2], obj[3], obj[4])
                    if dark: break

            if np.dot(normal,sun_dir)<0 or dark:
                continue
            color = color + obj_color*sun_color*np.dot(normal,sun_dir)
    if len(bulbs):
        for bulb in bulbs:
            bulb_dir, bulb_color = np.array(bulb[:3])-p, np.array(bulb[3:])
            d = np.linalg.norm(bulb_dir)
            bulb_dir = bulb_dir/d

            dark = False
            if obstacles:
                for obj in obstacles:
                    if obj[0] == 'sphere':
                        dark = bulbContactSphere(p,bulb_dir,obj[1],obj[2],d)
                    if obj[0] == 'plane':
                        dark = bulbContactPlane(p,bulb_dir,obj[1],obj[2],obj[3],obj[4],d)
                    if dark: break

            if np.dot(normal,bulb_dir)<0 or dark:
                continue
            color = color + obj_color*bulb_color*np.dot(normal,bulb_dir)/(d**2)
    return color


def rayPlane(ro, rd, a, b, c, d):
    normal = np.array([a,b,c])
    if a: p = np.array([-d/a,0,0])
    elif b: p = np.array([0,-d/b,0])
    else: p = np.array([0,0,-d/c])
    if np.dot(rd,normal) == 0: return np.inf
    t = np.dot((p-ro),normal)/np.dot(rd,normal)
    return t


def sunContactPlane(ro, rd, a, b, c, d):
    normal = np.array([a,b,c])
    if a: p = np.array([-d/a,0,0])
    elif b: p = np.array([0,-d/b,0])
    else: p = np.array([0,0,-d/c])
    if np.dot(rd,normal) == 0: return False
    t = np.dot((p-ro),normal)/np.dot(rd,normal)
    return t>0


def bulbContactPlane(ro, rd, a, b, c, d, bulbdist):
    normal = np.array([a,b,c])
    if a: p = np.array([-d/a,0,0])
    elif b: p = np.array([0,-d/b,0])
    else: p = np.array([0,0,-d/c])
    if np.dot(rd,normal) == 0: return False
    t = np.dot((p-ro),normal)/np.dot(rd,normal)
    return t>0 and np.linalg.norm(t*rd)<bulbdist


def invPlane(ro, rd, t, a, b, c, d, obj_color, suns, bulbs, obstacles):
    color = np.array([0,0,0])
    p = ro+t*rd
    normal = np.array([a,b,c])
    normal = normal/np.linalg.norm(normal)
    if len(suns):
        for sun in suns:
            sun_dir, sun_color = np.array(sun[:3]), np.array(sun[3:])
            sun_dir = sun_dir/np.linalg.norm(sun_dir)

            dark = False
            if obstacles:
                for obj in obstacles:
                    if obj[0] == 'sphere':
                        dark = sunContactSphere(p,sun_dir,obj[1],obj[2])
                    if obj[0] == 'plane':
                        dark = sunContactPlane(p,sun_dir,obj[1],obj[2],obj[3],obj[4])
                    if dark: break

            if np.dot(normal,sun_dir)<0 or dark:
                continue
            color = color + obj_color*sun_color*np.dot(normal,sun_dir)
    if len(bulbs):
        for bulb in bulbs:
            bulb_dir, bulb_color = np.array(bulb[:3])-p, np.array(bulb[3:])
            d = np.linalg.norm(bulb_dir)
            bulb_dir = bulb_dir/d

            dark = False
            if obstacles:
                for obj in obstacles:
                    if obj[0] == 'sphere':
                        dark = bulbContactSphere(p,bulb_dir,obj[1],obj[2],d)
                    if obj[0] == 'plane':
                        dark = bulbContactPlane(p,bulb_dir,obj[1],obj[2],obj[3],obj[4],d)
                    if dark: break

            if np.dot(normal,bulb_dir)<0 or dark:
                continue
            color = color + obj_color*bulb_color*np.dot(normal,bulb_dir)/(d**2)
    return color


def drawRaster(image, dim, objs, suns, bulbs, v):
    width, height = dim
    eye = np.array([0,0,0])
    forward = np.array([0,0,-1])
    right, up = np.array([1,0,0]), np.array([0,1,0])
    maxwh = max(width,height)
    collection = []
    for obj in objs:
        if obj[0] == 'sphere':
            c, r = np.array(obj[1:4]), obj[4]
            obj_color = np.array(obj[5:])
            collection.append(['sphere',c,r,obj_color])
        elif obj[0] == 'plane':
            a, b, c, d = obj[1:5]
            obj_color = np.array(obj[5:])
            collection.append(['plane',a,b,c,d,obj_color])
    
    for x in range(width):
        for y in range(height):
            coor = tuple([x,y])
            collision = []
            sx, sy = (2*x-width)/maxwh, (height-2*y)/maxwh
            ro, ray = eye, forward+sx*right+sy*up
            rd = ray/np.linalg.norm(ray)
            for obj in objs:
                if obj[0] == 'sphere':
                    c, r = np.array(obj[1:4]), obj[4]
                    obj_color = np.array(obj[5:])
                    t = raySphere(ro, rd, c, r)
                    if t != np.inf and t > 0:
                        collision.append(['sphere',t,c,r,obj_color])
                elif obj[0] == 'plane':
                    a, b, c, d = obj[1:5]
                    obj_color = np.array(obj[5:])
                    t = rayPlane(ro, rd, a, b, c, d)
                    if t != np.inf and t > 0:
                        collision.append(['plane',t,a,b,c,d,obj_color])
            if collision:
                # input(sorted(collision,key=lambda cols: cols[1]))
                obstacles = collection[:]
                final_obj = sorted(collision,key=lambda cols: cols[1])[0]
                selfobj = final_obj[:]
                selfobj.pop(1)
                obs_noself = []
                for element in obstacles:
                    if not objEqual(element,selfobj):
                        obs_noself.append(element)
                if final_obj[0] == 'sphere':
                    t, c, r, obj_color = final_obj[1:]
                    color = invSphere(ro,rd,t,c,r,obj_color,suns,bulbs,obs_noself)
                elif final_obj[0] == 'plane':
                    t,a,b,c,d,obj_color = final_obj[1:]
                    color = invPlane(ro,rd,t,a,b,c,d,obj_color,suns,bulbs,obs_noself)
                sRGBcolor = convertColor(color, v)
                image.im.putpixel(coor,sRGBcolor)

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