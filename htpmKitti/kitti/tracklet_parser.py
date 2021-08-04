#!/usr/bin/env python3

# Import main
import sys
import os
import numpy as np

# Import select
import htpmKitti.kitti.parseTrackletXML as xmlParser


# -------------------------------------------
def main(kittiDir,drive):
    xmlExtension    = 'tracklet_labels.xml'       
    labelExtension  = 'label_2' 
    timeExtension  = 'image_02/timestamps.txt'

    # Get velo to camera calibration matrix
    f       = open(kittiDir + '/calib_velo_to_cam.txt')
    lines   = f.readlines()
    # Get R matrix
    Rargs = lines[1].rstrip().split(' ')  
    Rargs.remove('R:')
    Rargs = [float(i) for i in Rargs]
    R = np.array([Rargs[0:3], Rargs[3:6], Rargs[6:9]])  

    # Get T matrix
    Targs = lines[2].rstrip().split(' ')
    Targs.remove('T:')
    Targs = [float(i) for i in Targs]
    T = np.array(Targs,ndmin=2) 

    # Get H matrix
    H = np.concatenate((R,np.transpose(T)),axis = 1)
    f.close

    # Load xml tree into python
    pyObjects   = xmlParser.parseXML(kittiDir + '/' + drive + '/' + xmlExtension) 

    # Convert xml tree to ros object list
    # rosObjects  = []

    # Get image sequence length
    n = len(open(kittiDir + '/' + drive + '/' + timeExtension ).readlines(  ))

    # Make dir if not exists
    if not os.path.isdir(kittiDir + '/' + drive + '/' + labelExtension):
        os.mkdir(kittiDir + '/' + drive + '/' + labelExtension)

    # Make object files
    for i in range(n):
        fileName = str(i)
        fileName = kittiDir + '/' + drive + '/' + labelExtension + '/' + fileName.zfill(6) + '.txt' 
        # Clear file if exists
        try:
            f = open(fileName, 'r+')
            f.truncate(0)
            f.close
        # Create file if not exists
        except:
            f = open(fileName,'w')

    # Get number of tracklets
    o = len(pyObjects)

    # Get tracklet indices
    for j in range(o):
        # Reset inter frame counter
        x = 0
        for k in range(pyObjects[j].firstFrame, pyObjects[j].firstFrame + pyObjects[j].nFrames ):
            # Open correct file
            fileNumber = str(k)
            fileName = kittiDir + '/' + drive + '/' + labelExtension + '/' + fileNumber.zfill(6) + '.txt' 
            f = open(fileName, 'a')
            
            # Add object information
            # Type        
            f.write(pyObjects[j].objectType + ' ')   
            # Truncation
            if ((pyObjects[j].truncs[x] >= 0) and (pyObjects[j].truncs[x] <= 1)):     
                f.write(str(pyObjects[j].truncs[x]) + ' ')              
            else: 
                f.write(str(2) + ' ')
            # Occlusion 
            if ((pyObjects[j].occs[x][0] >= 0) and (pyObjects[j].occs[x][0] <= 2)):   
                f.write(str(int(pyObjects[j].occs[x][0])) + ' ') 
            else:
                f.write(str(3) + ' ')       
            # Rotation along z (alpha)
            uvw = np.append(pyObjects[j].trans[x], 1)
            xyz = np.matmul(H,np.transpose(uvw))
            f.write(str("%.2f" % np.arctan2(xyz[0], xyz[2])) + ' ') 
            # Bounding box 
            h,w,l = pyObjects[j].size
            bbox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
            [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
            [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
            [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])
            # BBox placeholder
            f.write('0.0 0.0 0.0 0.0' + ' ')       
            # Size                              
            f.write(str("%.2f" % h) + ' ' + str("%.2f" % w) + ' ' + str("%.2f" %l) + ' ') 
            # Location
            f.write(str("%.2f" % h) + ' ' + str("%.2f" % w) + ' ' + str("%.2f" %l) + ' ') 
            # Rotation
            f.write(str("%.2f" % pyObjects[j].rots[x][2]))
            f.write('\n')
            
            f.close
            x = x+1
        

if __name__ == "__main__":
    if len(sys.argv)>1:
        kittiDir= sys.argv[1]
    else:
        kittiDir    = '/home/jim/HDDocuments/university/master/thesis/ROS/data/2011_09_26/'
    roadTypes   = ['road','city','residential']
    for i in range(0,len(roadTypes)):
        driveNames = sorted(os.listdir(os.path.join(kittiDir,roadTypes[i])))
        for j in range(0,len(driveNames)):
            print(driveNames[j])
            main(kittiDir,os.path.join(roadTypes[i],driveNames[j]))

# objectType = None
# size = None  # len-3 float array: (height, width, length)
# firstFrame = None
# trans = None   # n x 3 float array (x,y,z)
# rots = None    # n x 3 float array (x,y,z)
# states = None  # len-n uint8 array of states
# occs = None    # n x 2 uint8 array  (occlusion, occlusion_kf)
# truncs = None  # len-n uint8 array of truncation
# amtOccs = None    # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
# amtBorders = None    # None (n x 3) float array  (amt_border_l / _r / _kf)
# nFrames = None

# # Header header
# string type
# float32 truncated
# int8 occluded
# float32 alpha
# geometry_msgs/Quaternion bbox
# geometry_msgs/Vector3 dimensions
# geometry_msgs/Vector3 location
# float32 location_y


