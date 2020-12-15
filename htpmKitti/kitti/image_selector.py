#!/usr/bin/env python3

# Import main
import os
import random
import shutil
import sys
import numpy
import pickle



# Get drive folders
dataPath    = '/home/jim/HDDocuments/university/master/thesis/ROS/data/2011_09_26'
imagePath   = 'image_02/data'
labelPath   = 'label_2'
oxtsPath    = 'oxts/data'
roadTypes   = ['road', 'city', 'residential']

# Drive class for organizing data
class drive:
    def __init__(self, dataPath, roadType, driveName):
        self.driveName  = driveName.split('_')[4]
        self.roadType   = roadType 
        self.dataPath   = os.path.join(roadType, driveName)
        
        self.imagePath  = os.path.join(self.dataPath, imagePath) 
        self.labelPath  = os.path.join(self.dataPath, labelPath)
        self.oxtsPath   = os.path.join(self.dataPath, oxtsPath) 
        
        self.images     = sorted(os.listdir(os.path.join(dataPath, self.imagePath)))
        self.labels     = sorted(os.listdir(os.path.join(dataPath, self.labelPath)))
        self.oxts       = sorted(os.listdir(os.path.join(dataPath, self.oxtsPath)))
        
    def __str__(self):
        return 'Drive object with Name: ' + self.driveName + ', Type: ' + self.roadType + ', and Path: ' + self.dataPath
    
    def __repr__(self):
        return 'drive_' + str(self.driveName)
    

# Create empty lookup table for drives
drives      = [ [] for _ in range(len(roadTypes)) ]

# Get drives 
for i in range(0, len(roadTypes)):
    driveNames = sorted(os.listdir(os.path.join(dataPath, roadTypes[i])))
    
    for j in range(0, len(driveNames)):
        newDrive = drive(dataPath, roadTypes[i], driveNames[j])  
        drives[i].append(newDrive)

# Set random seed or fill in seedValue
# seedValue = random.randrange(sys.maxsize) 
seedValue = 6632973293414144069
random.seed(seedValue) 

# Setup random drives folder
folderName = 'test_images'
try:
    shutil.rmtree(os.path.join(dataPath, folderName))
except:
    pass
os.mkdir(os.path.join(dataPath, folderName))
os.mkdir(os.path.join(dataPath, folderName, 'image_02'))
os.mkdir(os.path.join(dataPath, folderName, 'image_02', 'data'))
os.mkdir(os.path.join(dataPath, folderName, 'label_2'))
os.mkdir(os.path.join(dataPath, folderName, 'oxts'))
os.mkdir(os.path.join(dataPath, folderName, 'oxts', 'data'))


# #################################### Random selection ###########################
# # Save seed
# seedFile = open(os.path.join(dataPath,folderName,'seed.txt'),'w')
# seedFile.write(str(seedValue))
# seedFile.close()

# # Get random images and put in folder
# driveList = []
# for n in range(0,210): 
#     # Get random road type
#     r       = random.randint(0, 2)
#     # Get random drive
#     rDrive  = random.choice(drives[r])
#     # Get random image  
#     rIndex  = random.randrange(len(rDrive.images))
#     rImage  = rDrive.images[rIndex]    
#     # Append image to random list
#     shutil.copyfile(os.path.join(dataPath,rDrive.imagePath,rImage), os.path.join(dataPath,folderName,'image_02','data',('%s.png' %n)))
#     # Get corresponding random label
#     rLabel  = rDrive.labels[rIndex]
#     # Append object labels to random list
#     shutil.copyfile(os.path.join(dataPath,rDrive.labelPath,rLabel), os.path.join(dataPath,folderName,'label_2',('%s.txt' %n)))
#     # Save drive to list
#     driveList.append([rDrive,rIndex])

# # Save drives for later examination
# driveFile = open(os.path.join(dataPath,folderName,'driveList.dat'),'wb')
# pickle.dump(driveList, driveFile, protocol = 2)
# driveFile.close()

# Example unload
# driveFile = open(os.path.join(dataPath,folderName,'driveList.dat'),'rb')
# driveList2 = pickle.load(driveFile)
# roadType0 = driveList2[3][0].roadType
# driveFile.close()
# print(roadType0)

# #################################### Uniform selection ############################

# Declare number of images total
total_frames = 210

# Declare empties
frame_indices = [[], [], []]
n_frames = [0, 0, 0]
all_image_paths         = [[], [], []]
all_label_paths         = [[],[],[]]
all_oxts_paths          = [[],[],[]]
uniform_image_selection = []
uniform_label_selection = []
uniform_oxts_selection  = []

# Loop over drives
for i in range(0, len(roadTypes)):
    previous_drive_frames = 0
    for j in range (0, len(drives[i])):
        if drives[i][j].roadType == roadTypes[i]:
            # Get number of frames per road type
            n_frames[i] = n_frames[i]+len(drives[i][j].images)
    frame_indices[i] = numpy.linspace(0, n_frames[i]-1, total_frames/len(n_frames), dtype = int)


for i in range(0, len(roadTypes)):
    for j in range(0, len(drives[i])):
        for k in range(0, len(drives[i][j].images)):
            all_image_paths[i].append(os.path.join(drives[i][j].imagePath, drives[i][j].images[k]))
            all_label_paths[i].append(os.path.join(drives[i][j].labelPath, drives[i][j].labels[k]))
            all_oxts_paths[i].append(os.path.join(drives[i][j].oxtsPath, drives[i][j].oxts[k]))

    for n in frame_indices[i]:  
        uniform_image_selection.append(all_image_paths[i][n])
        uniform_label_selection.append(all_label_paths[i][n])
        uniform_oxts_selection.append(all_oxts_paths[i][n])

# # Save seed
seedFile = open(os.path.join(dataPath,folderName,'seed.txt'),'w')
seedFile.write(str(seedValue))
seedFile.close()

# Shuffle list for participants
shuffle_list = list(zip(uniform_image_selection,uniform_label_selection,uniform_oxts_selection))
random.shuffle(shuffle_list)
uniform_image_selection,uniform_label_selection,uniform_oxts_selection = zip(*shuffle_list)


# Save image list
uniformFile = open(os.path.join(dataPath,folderName,'uniform_image_list.txt'),'w')

# Copy all relevant data
for n in range(0,len(uniform_image_selection)):
    uniformFile.writelines(str(uniform_image_selection[n])+'\n')
    shutil.copyfile(os.path.join(dataPath,uniform_image_selection[n]), os.path.join(dataPath,folderName,imagePath,('%s.png' %n)))
    shutil.copyfile(os.path.join(dataPath,uniform_label_selection[n]), os.path.join(dataPath,folderName,labelPath,('%s.txt' %n)))
    shutil.copyfile(os.path.join(dataPath,uniform_oxts_selection[n]), os.path.join(dataPath,folderName,oxtsPath,('%s.txt' %n)))

uniformFile.close()



