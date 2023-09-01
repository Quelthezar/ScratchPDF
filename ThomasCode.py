# The following program uses the the OpenCV library to detect particle trajectories of spherical particles in videos.
# The program was written by Thomas Long who can be reached at tlong2010@gmail.com
# The S update gives a more efficient pair counting algorithm
# The T update removes the "import cv2.cv as cv" when the version is not 2, and adds the particle radius as an adjustable setting
# The U update puts curve fitting into the python code with scipy.optimize
# The W update does aggregate counting with contours rather than circle detection
# The X update compares shapes and adds time shifted calculations
# The Y update adds parallel processing and cleans up the code
# The Z update adds a "nearest neighbor" calculation. And a trajectory calculation for the contours. Circle detection is removed. Will add pair collision/force constant fitting.
# The AA update removes redundant variables. Cleans code.
# The AB update changes from the threshold method of countour detection to the canny edge detection of contours
# The AC update adds trajectory processing for the contours
# The AD update adds Mean Squared Displacement calculations for each time
# The AF update adds magnetic dipole strength equation fists by splitting the trajectory processing into magnetic and Brownian regimes
# The AG update adds parallel
# The AK update removes "hierarchy" so that it works across OpenCV.2.4.* and OpenCV.3.0.*
# The AL update adds Polymer-type calculations and dipole calculations
# The AN update adds collision counting and fixes C5 calculations
#import pip
#sys.path.append('/scratch/thomas.long/virtualenv-15.1.0/test/Python-3.6.1/lib/python3.6/site-packages')
import sys
import cv2
if(cv2.__version__[0]=='2'): # if the version is 2.x.x
    import cv2.cv as cv # import the cv
    CV_CAP_PROP_FPS=cv.CV_CAP_PROP_FPS
    #CV_HOUGH_GRADIENT=cv.CV_HOUGH_GRADIENT
    CV_CAP_PROP_FRAME_COUNT=cv.CV_CAP_PROP_FRAME_COUNT #http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
    CV_CAP_PROP_POS_FRAMES=cv.CV_CAP_PROP_POS_FRAMES
    boxPoints=cv2.cv.BoxPoints
    #fourcc = cv2.cv.CV_FOURCC('D','I','V','X') #FourCC code for AVI format
else: # if version 3.x.x
    CV_CAP_PROP_FPS=cv2.CAP_PROP_FPS #the number of frames per second, 5 is the "get" for avi videos, might be different for other types, 2 is 1/fps, 3 and 4 are height and width, 7 is frame count
    #CV_HOUGH_GRADIENT=cv2.HOUGH_GRADIENT
    CV_CAP_PROP_FRAME_COUNT=cv2.CAP_PROP_FRAME_COUNT
    CV_CAP_PROP_POS_FRAMES=cv2.CAP_PROP_POS_FRAMES
    boxPoints=cv2.boxPoints
    #fourcc = cv2.VideoWriter_fourcc('D','I','V','X') #FourCC code for AVI format
import numpy as np
import math
import os.path
import time
import random
import csv
print(cv2.__version__) # prints the opencv version
startTime=time.time()
CalculateTrajectories=1 # Set to 1 in order to do the extra work of calculating trajectories
RunInParallel=0 # tells the system to run in Parallel if equals 1


xrange=range # xrange is not defined on python3

if len(sys.argv)==2:
    myfile = open(sys.argv[1], 'r')
    cap = cv2.VideoCapture(sys.argv[1]) #takes the system argument, input the command line
    filename=sys.argv[1]
DistanceBetweenMovesInitial=8
ThresholdInitial=150
Field=1.570796326 #math.pi*0.5 #orientation of magnetic field in radians, make it between 0 and pi
MeasuredField=Field #orientation of measured magnetic field in radians
SlopeField=500 # determined by the "Field Variable"
if(abs(math.pi/2-Field)>0.0001): # if the angle is pi/2, the angle is infinite and the slope will be set to 100
    SlopeField=math.tan(Field)
FinalFrame=int(cap.get(CV_CAP_PROP_FRAME_COUNT)-1) #6
FrameSampleRate=FinalFrame/10 #the rate of sampling of the frams, eg. 10 will analyze every tenth frame
LogScaleSampleRate=1 # The sampling rate multiplier so frame(n+1)=(frame(n)+FrameSampleRate)*LogScaleSampleRate
PeOneSec=2 # the time it takes for particles at Peclet number 1 to aggregate
fps=cap.get(CV_CAP_PROP_FPS) # http://www.learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/
MedianBlurSetting=7
print("cap.get(CV_CAP_PROP_FRAME_COUNT)",cap.get(CV_CAP_PROP_FRAME_COUNT))
ffmpegCommand="ffmpeg -f image2 -pattern_type glob -nostats -loglevel 0 -i '"+filename
if os.path.isfile(str(filename)+'.settings.csv'):
    print('Loading preset setting from file: '+str(filename)+'.settings.csv \n\n')
    f = open(filename+'.settings.csv','r') #open a text file to read
    lines=f.readlines()
    # search for a variable name in a row in the file, the variable is right under that so add 1 to that, then set the variable of interest equal to that
    DistanceBetweenMovesInitial=int(lines[lines.index('Distance Between Moves\n')+1])
    ParRMicrons=float(lines[lines.index('Particle Radius (Microns)\n')+1])
    ThresholdInitial=int(lines[lines.index('Threshold\n')+1])
    MicronPerPix=float(lines[lines.index('Microns Per Pixel\n')+1])
    Field=float(lines[lines.index('Field (radians)\n')+1])
    fps=float(lines[lines.index('fps\n')+1])
    AreaPerParticle=float(lines[lines.index('Area Per Singlet(pix^2)\n')+1])
    PeOneSec=float(lines[lines.index('Time to Aggregate from Pe One in Seconds\n')+1])
    FinalFrame=int(lines[lines.index('FinalFrame\n')+1])
    FrameSampleRate=int(lines[lines.index('FrameSampleRate\n')+1])
    RunInParallel=int(lines[lines.index('RunInParallel\n')+1])
    MedianBlurSetting=int(lines[lines.index('MedianBlurSetting\n')+1])
    f.close() # close the file
else:
    ParRMicrons=1.2
    MicronPerPix=0.7501
    AreaPerParticle=40.0
    filename_split=filename.split("_") # split the file name to search for settings information
    for iii in xrange(0,len(filename_split)):
        if(filename_split[iii][-7:]=="microns"):
            ParRMicrons=float(filename_split[iii][:-7])/2.0
        elif(filename_split[iii][-1:]=="x"):
            print(int(filename_split[iii][:-1]))
            magnification=int(filename_split[iii][:-1])
            if(magnification==5):
                MicronPerPix=0.7501
                AreaPerParticle=40.0
            elif(magnification==10):
                MicronPerPix=0.4714
                AreaPerParticle=30.0
            elif(magnification==20):
                MicronPerPix=0.2347

    ParRMicrons = float(raw_input("\n Please input the particle size in microns (default="+str(ParRMicrons)+"): ") or str(ParRMicrons)) #Particle Radius in microns
    ParRMicrons=float(ParRMicrons)
    # Microscope micrometer (old 5x): 0.7501 micron/pixel
    # Microscope micrometer (new 10x): 0.4714 micron/pixel
    # Microscope micrometer (new 20x): 0.2347 micron/pixel
    MicronPerPix = float(raw_input("\n Hello, user.\n Please input the Microns Per Pixels (old_5x=0.7501, new_10x=0.4714, new_20x=0.2347, default="+str(MicronPerPix)+"): ") or str(MicronPerPix))
    #AreaPerParticle = math.pi*ParRMicrons**2
    AreaPerParticle = float(raw_input("\n Please input the Area Per Particle in Pixels (default="+str(AreaPerParticle)+"): ") or str(AreaPerParticle))
    MedianBlurSetting = float(raw_input("\n Please input the Median Blur Setting in Pixels (default=7): ") or "7")
    fps = float(raw_input("\n Hello, user.\n Please input the frames per second in the video, default="+str(fps)+": ") or str(fps))
    FinalFrame = float(raw_input("\n Please input the FinalFrame to analyze, default="+str(FinalFrame)+": ") or str(FinalFrame))
    FrameSampleRate = float(raw_input("\n Please input the FrameSampleRate, default="+str(1+int(FinalFrame/10))+": ") or str(1+int(FinalFrame/10)))
    PeOneSec=ParRMicrons**3/0.2451
    PeOneSec = float(raw_input("\n Please input the time it takes two particles at Pe number 1 to aggregate, default="+str(PeOneSec)+": ") or str(PeOneSec))
    RunInParallel = float(raw_input("\n Please input 1 if you want to run this in parallel, default="+str(RunInParallel)+": ") or str(RunInParallel))

FinalFrame=int(FinalFrame)
FrameSampleRate=int(FrameSampleRate)
RunInParallel=int(RunInParallel)
PeOneFrames=int(PeOneSec*fps) # the time (in frames) it takes for particles at Peclet number 1 to aggregate
if(PeOneFrames>FinalFrame/2): # prevent the Peclet time from being greater than the final time, this prevents errors later on
    PeOneFrames=FinalFrame/2
PeOneSamples=int(float(PeOneFrames)/float(FrameSampleRate))
MedianBlurSetting=int(MedianBlurSetting)
if(PeOneSamples==0):
    PeOneSamples=1
PeOneSec=PeOneSamples*float(FrameSampleRate)/fps
ret, frame = cap.read() #look at the first image
frameno=0
#force the images and their variables to be global variables
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
heightV , widthV , layers =  frame.shape
ParRPixels=ParRMicrons/MicronPerPix
Viscosity=8.9*(10**-10) # Viscosity of water in kg/(microns*s) http://www.wolframalpha.com/input/?i=Viscosity+of+water+in+kg%2F(micron*s)
Vacuum_permeability=1.257 # mu0 [=] kg*microns*s^-2*A^-2 # http://www.wolframalpha.com/input/?i=vacuum+permeability+in+kg*microns*s%5E-2*A%5E-2
Fivemu0__4pisq_mu_r = 5*Vacuum_permeability/(4*(math.pi**2)*Viscosity*ParRMicrons) # the product of the dipole moments times this is the C5 constant in the equation: r(t) = (C5 (t0-t))^1/5 # m1=sqrt(C5/Fivemu0__4pisq_mu_r) for singlets # http://www.wolframalpha.com/input/?i=5*Vacuum+permeability%2F(4*(3.14159**2)*Viscosity+of+water*1.2+microns)
print("Viscosity",Viscosity,"kg/(microns*s)")
print("Vacuum_permeability",Vacuum_permeability,"kg*microns*s^-2*A^-2")
print("Fivemu0__4pisq_mu_r",Fivemu0__4pisq_mu_r,"microns/(s*A^2)")
DSE=0.2451/ParRMicrons # Diffusion Coefficient from Stokes-Einstein equation in Micrometers^2 Per second # http://www.wolframalpha.com/input/?i=(1.38*10%5E-23+J)*298%2F(6*pi*viscosity+of+water) # http://www.wolframalpha.com/input/?i=2.451*10%5E-19+m%5E3%2Fs+in+micron%5E3%2Fs
TauR=6.115*ParRMicrons**3 #Rotational Diffusion Time (s)
tBrown=ParRMicrons**2/DSE # the characteristic Brownian Time
imageCount=0 # the way to name the images sequentially
#writing settings to files
f = open(filename+'.settings.csv','w') #open a text file to write in
f.write('Distance Between Moves'+'\n'+str(DistanceBetweenMovesInitial)+'\n')
f.write('Particle Radius (Microns)'+'\n'+str(ParRMicrons)+'\n')
f.write('Threshold'+'\n'+str(ThresholdInitial)+'\n')
f.write('Microns Per Pixel'+'\n'+str(MicronPerPix)+'\n') # write the conversion factor
f.write('Field (radians)'+'\n'+str(Field)+'\n') # write the angle of the magnetic Field
f.write('fps'+'\n'+str(fps)+'\n') # write the angle of the magnetic Field
f.write('Area Per Singlet(pix^2)'+'\n'+str(AreaPerParticle)+'\n') # write the conversion factor
f.write('Time to Aggregate from Pe One in Seconds'+'\n'+str(PeOneSec)+'\n') # the time it takes to aggregate from a distance that has peclet number of 1
f.write('FinalFrame'+'\n'+str(int(FinalFrame))+'\n')
f.write('FrameSampleRate'+'\n'+str(int(FrameSampleRate))+'\n')
f.write('RunInParallel'+'\n'+str(int(RunInParallel))+'\n')
f.write('MedianBlurSetting'+'\n'+str(int(MedianBlurSetting))+'\n')
f.close() # close the file



os.system("rm "+filename+".*BoxLocation.csv && rm "+filename+".*MassParticles.csv && rm "+filename+".*length.csv") # remove extra save files

def CheckContour( contours, frame, hierarchy, framenoI ): #check the contours on a frame and output the frame with the contours drawn on according to size
    DoubletAngle = []
    for check in xrange(0,len(contours)):
        cnt = contours[check] # a particular contour
        rect = cv2.minAreaRect(cnt)
        box = boxPoints(rect)
        box = np.int0(box)
        #print("check,framenoI:",check,framenoI)
        if(len(cnt)<6): # if there are too few points, it is ignored
            pass
        else:
            ellipse=cv2.fitEllipse(cnt)
            hull = cv2.convexHull(cnt) #hull has no concavity
            cnt=hull
            epsilon = 0.01*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            area = cv2.contourArea(approx)# area enclosed by the contour can be used to find the aggregate size
            rect = cv2.minAreaRect(cnt)
            box = boxPoints(rect)
            box = np.int0(box)
            extent=0.00001
            mask = np.zeros(gray.shape,np.uint8) #make an empty mask of the size of the image
            cv2.drawContours(mask,[cnt],0,255,-1)
            smallSideSqr=min([ (box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2 , (box[0][0]-box[2][0])*(box[0][0]-box[2][0])+(box[0][1]-box[2][1])*(box[0][1]-box[2][1]) , (box[0][0]-box[3][0])*(box[0][0]-box[3][0])+(box[0][1]-box[3][1])*(box[0][1]-box[3][1])] )
            smallSideSqr=max([smallSideSqr,0.000001])
            # finds the shortest side to use in aspect ratio calculation. Will not allow 0
            smallSide=math.sqrt(smallSideSqr)
            circleCount=area/AreaPerParticle#(math.pi*ParRPixels**2)
            maxYpoint= cnt[0][0]# the highest y point
            minYpoint= cnt[0][0]# the lowest y point
            maxXpoint= cnt[0][0]# the leftmost x point
            minXpoint= cnt[0][0]# the rightmost x point
            for iii in xrange(1,len(cnt)):
                if(maxXpoint[0]<cnt[iii][0][0]):
                    maxXpoint=cnt[iii][0]
                if(minXpoint[0]>cnt[iii][0][0]):
                    minXpoint=cnt[iii][0]
                if(maxYpoint[1]<cnt[iii][0][1]):
                    maxYpoint=cnt[iii][0]
                if(minYpoint[1]>cnt[iii][0][1]):
                    minYpoint=cnt[iii][0]
            HeightAgg=float(abs(maxYpoint[1]-minYpoint[1])) #The height of the aggregate
            if(circleCount<1.5):
                HeightAgg=2*ParRPixels
            if(min(ellipse[1])>30 or min(ellipse[1])<ParRPixels*0.75 or smallSide>8*8*ParRPixels):
                pass
            elif(area<-MinArea): #delete if less than the minimum area, or inside another contours
                pass
                #cv2.putText(frame,str(extent)+" area"+str(area/MinArea), (cnt[0][0][0],cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
                pass
            else:
                open(filename + "." + str(framenoI) + '.BoxLocation.csv', 'a').write(',')  # have a space for each different particle corners
                for boxiii in xrange(len(box)): # loop through the 4 corners of the box
                    open(filename + "." + str(framenoI) + '.BoxLocation.csv', 'a').write(str(box[boxiii][0]) + ',' + str(box[boxiii][1]) + ',')  # 8 columns for the 4 corners of the enclosing box
                cv2.drawContours(frame, [cnt], -1, (255,100,255), 1)
                cv2.drawContours(frame, [box], -1, (255,100,255), 1)
                #cv2.putText(frame,str(check)+","+str(), (cnt[0][0][0],cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
                cv2.putText(frame,str(int(circleCount+0.5)), (cnt[0][0][0],cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
                #cv2.putText(frame,str(circleCount+0.5), (cnt[0][0][0],cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                M = cv2.moments(cnt) # http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html#gsc.tab=0
                if(M['m00']==0): M['m00']+=0.1 # avoid "divide by 0" error
                cx = int(M['m10']/M['m00']) # centroid x
                cy = int(M['m01']/M['m00']) # centroid y
                if(framenoI==0): # do some calculations only on the first frame
                    AverageDis = 0
                    averageDistance=heightV+widthV # make the average distance a large number so everything else is a minimum
                    for mmm in xrange(len(contours)):
                        MB = cv2.moments(contours[mmm])
                        if(MB['m00']>0 and mmm != check and len(contours[mmm])>6 and cv2.contourArea(contours[mmm])>MinArea): # make sure the paired particle is a particle
                            cxB = int(MB['m10']/MB['m00']) # centroid x
                            cyB = int(MB['m01']/MB['m00']) # centroid y
                            if(math.sqrt((cxB-cx)**2+(cyB-cy)**2)<averageDistance):
                                averageDistance=math.sqrt((cxB-cx)**2+(cyB-cy)**2)
                                cntPair=[cxB,cyB]
                    #cv2.putText(frame,str(int(averageDistance)), (cnt[0][0][0],cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
                    #                    cv2.line(frame,(cntPair[0],cntPair[1]),(cx,cy),(0,255,0),1)
                    AverageDis=((averageDistance+check*AverageDis)/(1+check)) # running average of the average distance between nearest neighbors
                open(filename + "." + str(framenoI) + '.length.csv', 'a').write(str((HeightAgg/math.cos(Field-0.5*math.pi))/(2*ParRPixels)) + ",")
                open(filename + "." + str(framenoI) + '.MassParticles.csv', 'a').write(','+str(circleCount))
                #                cv2.putText(frame,str(round(largeSide/ParRPixels,1)), (cnt[0][0][0],cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
                if(circleCount<1.5): cv2.drawContours(frame, [cnt], -1, (0,255,0), 1) # draw singlet contours green
                if(circleCount>=1.5 and circleCount<2.5): #doublets

                    side01=math.sqrt((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)
                    side12=math.sqrt((box[2][0]-box[1][0])**2+(box[2][1]-box[1][1])**2)
                    if(side01 > side12): # if 0,1 point is the long side, measure it as the angle of interest
                        #cv2.line(frame,( box[0][0], box[0][1]),(box[1][0], box[1][1]),(255,0,0),2) # draw doublet line to compare to field
                        #cv2.line(frame,( box[0][0], box[0][1]),( int(box[0][0]+50*math.cos(Field)), int(box[0][1]-50*math.sin(Field))),(0,255,255),1) # draw magnetic field line
                        if(abs(box[0][0]-box[1][0])>0.00001):
                            slopeDoublet=float(box[0][1]-box[1][1])/float(box[0][0]-box[1][0]) # slope of the side01 line
                        else:
                            slopeDoublet=1000.0
                    else: # 1,2 is the long side and should be used to measure the angle with the magnetic field
                        #cv2.line(frame,( box[2][0], box[2][1]),(box[1][0], box[1][1]),(255,0,0),2) # draw doublet line to compare to field
                        #cv2.line(frame,( box[1][0], box[1][1]),( int(box[1][0]+50*math.cos(Field)), int(box[1][1]-50*math.sin(Field))),(0,255,255),1) # draw magnetic field line
                        if(abs(box[2][0]-box[1][0])>0.00001):
                            slopeDoublet=float(box[2][1]-box[1][1])/float(box[2][0]-box[1][0]) # slope of the side01 line
                        else:
                            slopeDoublet=1000.0
                    slopeDoublet=-1*slopeDoublet
                    if((slopeDoublet*SlopeField)>-0.99 or (slopeDoublet*SlopeField)<-1.01):
                        DoubletAngle.append(math.atan(abs((slopeDoublet-SlopeField)/(1+slopeDoublet*SlopeField))))# http://planetmath.org/anglebetweentwolines
                    else:
                        DoubletAngle.append(math.pi/2)
                    #cv2.drawContours(frame, [cnt], -1, (0,0,255), 1) # draw doublet contours red
    if(len(DoubletAngle)>0): open(filename+'.doublet_angle.csv','a').write('\n'+str(framenoI/fps)+','+str(np.mean(DoubletAngle)*180/math.pi)+','+str(np.median(DoubletAngle)*180/math.pi)+',')
    return 0

FrameAreaPixels=heightV*widthV
FrameAreaMicrons=FrameAreaPixels*MicronPerPix**2
FrameAreaRadii=FrameAreaMicrons/(ParRMicrons**2)
MinArea=3.14159*(ParRPixels*0.2)**2
radius=ParRPixels # The average radius
AreaOfPar=3.14159*radius*radius # The average particle area
groups=[] # groups will be analyzed to fit the smoulkowski equations
print('Running Processor')
f = open(filename+'.doublet_angle.csv','w') #open a text file to write double angles in
f.write('Time(seconds), Average Angle, Median Angle')
f.close() # close the simple aggregates file

capE=[None]*int(FinalFrame)

FinalFrameNumber=0

def processAggregates(framenoI):
    print(' seconds processed: '+str(round(int(framenoI)/fps,1))+' out of: '+str(round(int(FinalFrame)/fps,1))+', approximate time remaining: '+str(round((int(FinalFrame)-int(framenoI))*20,0))+'s')
    kernel = np.ones((2,2),np.uint8) # kernel for erorion/dilation # https://en.wikipedia.org/wiki/Dilation_%28morphology%29 # http://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them
    capE[framenoI] = cv2.VideoCapture(filename)
    capE[framenoI].set(CV_CAP_PROP_POS_FRAMES,framenoI)
    ret, frame = capE[framenoI].read()
    capE[framenoI]=[None]
    #capB = cv2.VideoCapture(filename)
    #capB.set(CV_CAP_PROP_POS_FRAMES,framenoI)
    #ret, frame = capB.read()
    cv2.imwrite(filename+'.simple.'+(str(int(framenoI)).zfill(6))+'.png',frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img2 = gray
    img2 = cv2.medianBlur(img2,MedianBlurSetting) #was 5
    #img2 = fgbg.apply(img2) # background subtraction
    #img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
    cimg2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    cannySetting=17 # first setting, the second setting will be 3 times this, per canny reccomendation
    edges = cv2.Canny(cimg2,27,60) #edges

    #edges = cv2.dilate(edges,kernel,iterations = 1)
    #edges = cv2.erode(edges,kernel,iterations = 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    #contourshierarchy=[None,None]
    img_contours_hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # output is different for different versions of opencv # http://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html
    hierarchy = img_contours_hierarchy[-1]
    contours = img_contours_hierarchy[-2]
    maskEdge = frame*0
    for iii in xrange(0,len(contours)):
        cv2.drawContours(maskEdge, contours, iii, (255,255,255), -1) #fill in the edges
    maskEdge = cv2.cvtColor(maskEdge, cv2.COLOR_BGR2GRAY)
    img_contours_hierarchy = cv2.findContours(maskEdge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = img_contours_hierarchy[-1]
    contours = img_contours_hierarchy[-2]
    cv2.imwrite(filename+'.Edges.'+(str(int(framenoI)).zfill(6))+'.png',maskEdge)
    #for hhh in xrange(0,len(hierarchy[0])):
    #    print(hierarchy[0][hhh])
    #hierarchy=hierarchy[0,:,3].tolist() # convert the hierarchy to a list of numbers, will make it possible to delete them from the list
    DummyReturn=CheckContour( contours, frame, hierarchy, framenoI) #check the contours
    #cv2.line(frame,( widthV/2, heightV/2),(int(50*math.cos(Field)+widthV/2), int(heightV*0.5-50*math.sin(Field))),(0,255,255),1) # draw magnetic field line
    #cv2.putText(frame,str(int(Field*180/math.pi))+'deg', (widthV/2, heightV/2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    cv2.imwrite(filename+'.Counting.'+(str(int(framenoI)).zfill(6))+'.png',frame)
    FinalFrameNumber = int(framenoI)
    #ContAggCount=[int(framenoI-2)/fps]+ContAggCount
    return 0




if(RunInParallel): #if 0 run serial, if 1 run parallel
    from joblib import Parallel, delayed # http://blog.dominodatalab.com/simple-parallelization/
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    num_cores=3
    inputs = range(0, FinalFrame, FrameSampleRate)  # the range of frames
    resultsContAggCount=Parallel(n_jobs=num_cores)(delayed(processAggregates)(framenoI) for framenoI in inputs)
else:
    while(1):
        resultsContAggCount=processAggregates(frameno) #use contours to count the aggregates
        frameno=int((frameno+FrameSampleRate)*LogScaleSampleRate)
        if frameno+1>FinalFrame: #close the video if the user asks for the video to quit, or if the video reaches its maximum frame rate
            break

open(filename + '.length.csv', 'w').write('')
open(filename + '.BoxLocation.csv', 'w').write('Time(seconds),,Corner1-x,Corner1-y,Corner2-x,Corner2-y,Corner3-x,Corner3-y,Corner4-x,Corner4-y\n')
open(filename + '.MassParticles.csv', 'w').write('Time(seconds),MassPar1,MassPar2,MassPar3,...')
open(filename + '.simple_aggregates.csv', 'w').write('Time(seconds),Singlets,Doublets,Triplets,...')
open(filename + '.aggregates_concentration.csv', 'w').write('Time(seconds),Singlets,Doublets,Triplets,...')
#open(filename + '.NCCount.csv', 'w').write('Time(seconds),Number of Chains (par/r^2)')
open(filename + '.chainCount_concentration.csv', 'w').write('Time(t/t*),Number of Chains (chain/r^2), NumberAverage, MassAverage, ZAverage, ViscosityAverage, PDI')
open(filename+'.average_length.csv','w').write('Time(t/tBrown), Average_Length_(diameter), STD_Length_(diameter), Slope_Log_Log')

fout=open(filename + '.length.csv',"a") # merge length files
gout=open(filename + '.BoxLocation.csv',"a") # merge Box Location files
hout=open(filename + '.MassParticles.csv',"a") # merge Mass Particles files
iout=open(filename + '.simple_aggregates.csv',"a")
jout=open(filename + '.aggregates_concentration.csv',"a")
#kout=open(filename + '.NCCount.csv',"a")
mout=open(filename + '.chainCount_concentration.csv',"a")
groups=[None] * int(math.ceil(FinalFrame*1.0/FrameSampleRate))
for num in range(0,int(FinalFrame),FrameSampleRate):  #cap.get(CV_CAP_PROP_FRAME_COUNT)-1)
    for line in open(filename+"."+str(num)+".length.csv"):
        LengthListing=line.split(",")[:-1]
        LengthListing=[float(i) for i in LengthListing]
        open(filename + '.average_length.csv', 'a').write('\n' + str(num*1.0/(fps*tBrown)) + ',' + str(np.mean(LengthListing)) + ',' + str(np.std(LengthListing)))
        fout.write(line+"\n")
    for line in open(filename+"."+str(num)+".BoxLocation.csv"):
        gout.write(str(num*1.0/fps)+","+line+"\n")
    for line in open(filename+"."+str(num)+".MassParticles.csv"):
        hout.write("\n"+str(num*1.0/fps)+line)
        MassParticlesList = line.split(",")[1:] # the list of all the particle masses for the frame
        MassParticlesList = [float(i) for i in MassParticlesList]
        MassParticlesCount = np.zeros(1000) # The count of the masses, used for concentration of the chain types
        for iii in xrange(0, len(MassParticlesList)): # loop through the particles
            MassParticlesCount[int(round(MassParticlesList[iii]))]+=1
        MassParticlesCount[1]+=MassParticlesCount[0] # make all zero sized particles equal to 1
        MassParticlesCount=MassParticlesCount[1:] # get rid of the zero counter
        groups[int(num/FrameSampleRate)]=[(num*1.0/fps)]+(MassParticlesCount.tolist())[:20]
        iout.write("\n" + str(num * 1.0 / fps)+"," + str(MassParticlesCount.tolist())[1:-1])
        jout.write("\n" + str(num * 1.0 / (fps*tBrown)) + "," + str((MassParticlesCount/FrameAreaRadii).tolist())[1:-1])

        ChainConc=(sum(MassParticlesCount) / FrameAreaRadii).tolist()
        NumberAverage=np.dot(MassParticlesCount,np.arange(1,1000))/sum(MassParticlesCount) # https://en.wikipedia.org/wiki/Molar_mass_distribution
        MassAverage=np.dot(MassParticlesCount,np.square(np.arange(1,1000)))/np.dot(MassParticlesCount,np.arange(1,1000))
        ZAverage=np.dot(MassParticlesCount,np.power(np.arange(1,1000),3))/np.dot(MassParticlesCount,np.square(np.arange(1,1000)))
        aaa=2
        ViscosityAverage=(np.dot(MassParticlesCount,np.power(np.arange(1,1000),aaa+1))/np.dot(MassParticlesCount,np.arange(1,1000)))**(1.0/aaa)
        PDI=MassAverage/NumberAverage # https://en.wikipedia.org/wiki/Step-growth_polymerization#PDI
        #kout.write("\n" + str(num * 1.0 / (fps*tBrown)) + "," + str((sum(MassParticlesCount) / FrameAreaRadii).tolist())[1:-1])
        mout.write("\n" + str(num * 1.0 / (fps*tBrown)) + "," + str(ChainConc) + "," + str(NumberAverage) +"," +str(MassAverage) + "," + str(ZAverage) + "," + str(ViscosityAverage)+ "," + str(PDI) )
fout.close()
gout.close()
hout.close()
iout.close()
jout.close()
#kout.close()
mout.close()
os.system("rm "+filename+".*.length.csv")
os.system("rm "+filename+".*.BoxLocation.csv")
os.system("rm "+filename+".*.MassParticles.csv")
os.system(ffmpegCommand+".Counting.*.png' -y "+filename+".Counting.avi") # https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence#Making_an_Image_Sequence_from_a_video
os.system(ffmpegCommand+".Counting.*.png' -y "+filename+".Counting.gif") # https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence#Making_an_Image_Sequence_from_a_video
os.system(ffmpegCommand+".Edges.*.png' -y "+filename+".threshold.gif")
os.system(ffmpegCommand+".simple.*.png' -y "+filename+".simple.avi")
os.system(ffmpegCommand+".simple.*.png' -y "+filename+".simple.gif")
os.system("ffmpeg -framerate 1 -i 'concat:"+filename+".simple.000000.png|"+filename+".Counting.000000.png' -y "+filename+".FirstFrame.gif")





print('Processing Complete. Data files and processed video saved in current directory. \n')


#writing data to files
calculationsf = open(filename+'.calculations.csv','w') #open a text file to write in
TZeroList=[float(iii) for iii in groups[0][1:]]
nParticles=np.dot(TZeroList,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
SurfaceConcentrationPpMS= nParticles/(heightV*widthV*(MicronPerPix)*(MicronPerPix))#particle/micron^2
SurfaceAreaFraction=SurfaceConcentrationPpMS*3.14159*ParRMicrons**2

#print("groups",groups)
#groups = filter(None, groups) # remove "None"s that are left over
#print("groups",groups)

FrameAreaPixels=heightV*widthV
FrameAreaMicrons=FrameAreaPixels*MicronPerPix**2
FrameAreaRadii=FrameAreaMicrons/(ParRMicrons**2)
print("Area Pixels: ",FrameAreaPixels," , Area Microns: ",FrameAreaMicrons," , Area Radii: ",FrameAreaRadii)
print("Group List at time zero: ",TZeroList)
print("Number of Particles at time zero: ",nParticles)
print("Number of Chains at time zero: ",sum(TZeroList))
print("Surface Area Concentration: ",nParticles*math.pi/FrameAreaRadii)
print("Particles/Area: ",nParticles/FrameAreaRadii)
print("Singlets/Area: ",TZeroList[0]/FrameAreaRadii)
print("Chains/Area: ",sum(TZeroList)/FrameAreaRadii)

from scipy.integrate import odeint
from scipy import integrate
from scipy.optimize import fmin
Td=np.array([row[0] for row in groups])/tBrown# time
Oned=np.array([row[1] for row in groups])/FrameAreaRadii # singlets
Twod=np.array([row[2] for row in groups])/FrameAreaRadii # doublets
Threed=np.array([row[3] for row in groups])/FrameAreaRadii # triplets

def equation(par,initial_cond,start_t,end_t,incr): #http://adventuresinpython.blogspot.ca/2012/08/fitting-differential-equation-system-to.html
    #-time-grid-----------------------------------
    t  = np.linspace(start_t, end_t,incr)
    #differential-eq-system----------------------
    def funct(y,t):
        #K11,K12,K13,K22,K23,K33=par
        K11,K12,K13,K22,K23,K33=[abs(xxx) for xxx in par]
        # the equations
        f1 = - K11*y[0]**(4/2) - K12*y[1]*y[0]**(2/2) - K13*y[2]*y[0]**(2/2)
        f2 = 0.5*K11*y[0]**(4/2)-K12*y[1]*y[0]**(2/2)-K22*y[1]*y[1]-K23*y[1]*y[2]
        f3 = K12*y[1]*y[0]**(2/2)-K13*y[2]*y[0]**(2/2)-K23*y[1]*y[2]-K33*y[2]*y[2]
        return [f1, f2, f3]
    #integrate------------------------------------
    ds = integrate.odeint(funct,initial_cond,t)
    return (ds[:,0],ds[:,1],ds[:,2],t)
#2.Set up Info for Model System
# model parameters
guess=1.0
if(Oned[0]>0): guess=1/(Oned[0]*Td[-1]) # use second order reaction rate to guess the k value
if(Oned[0]>0 and Oned[-1]>0): guess=((1/(Oned[-1]))-(1/(Oned[0])))/Td[-1] # use second order reaction rate to guess the k value
[K11,K12,K13,K22,K23,K33]=[guess,guess,guess,guess,guess,guess]
rates=(K11,K12,K13,K22,K23,K33)
# model initial conditions
y0=[Oned[0],Twod[0],Threed[0]] #initial conditions
# model steps
start_time=0.0
end_time=Td[-1] #final time
intervals=len(Td) #number of sections to divide the fit line
mt=np.linspace(start_time,end_time,intervals)
# model index to compare to data
findindex=lambda x:np.where(mt>=x)[0][0]
mindex=map(findindex,Td)
#3.Score Fit of System
def score_equation(parms):
    #a.Get Solution to system
    F1,F2,F3,T=equation(parms,y0,start_time,end_time,intervals)
    #c.Score Difference between model and data points
    ss=lambda data,model:((data-model)**2).sum() #sum the residuals
    return ss(Oned,F1[mindex])+ss(Twod,F2[mindex])+ss(Threed,F3[mindex]) #sum the residuals because we want to minimize them
#4.Optimize Fit
#fit_score=score_equation(rates)
#answ=fmin(score_equation,(rates),full_output=1,maxiter=1000) # fmin minimizes the score_equation function by changing the rates # http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.fmin.html
#K11,K12,K13,K22,K23,K33=answ[0] # the best rates, the best score is at answ[1]
newrates=(K11,K12,K13,K22,K23,K33)
newratesAbs=(abs(K11),abs(K12),abs(K13),abs(K22),abs(K23),abs(K33))
#5.Generate Solution to System
F1,F2,F3,T=equation(newrates,y0,start_time,end_time,intervals)
#open(filename+".Smoluchowski_constants.csv", "w").write('K11,K12,K13,K22,K23,K33\n'+str(newratesAbs)[1:-1]) #print constants to file




#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(T,F1,'b-',Td,Oned,'gx',T,F2,'r-',Td,Twod,'yx',T,F3,'c-',Td,Threed,'kx')
#plt.legend(('Singlets Fit','Singlets Data','Doublets Fit','Doublets Data','Triplets Fit','Triplets Data'))
#plt.xlabel('Time (t/tBrown)')
#plt.ylabel('Amount of Chain(par/r^2)')
#plt.savefig(filename+'.plot_Smoluchowski.png') #print the plot


#tMinus=(nParticles-TZeroList[0])/(K11*nParticles*TZeroList[0])
tMinus=0
TM1,TM2,TM3,T=equation(newrates,[nParticles,0,0],start_time,end_time,intervals) # A shifted function that starts at the real time zero
for ttt in xrange(0,len(TM1)): #finds the time when all particles were singlets, will be 0 if the time is farther than the video length
    if(TM1[ttt]<=Oned[0]): # when the shifted time singlet concentration equals the actual initial concentration, make this time tminus and break the loop to continue
        tMinus=T[ttt]
        break
print("Time when all particles were singlets: ",tMinus)






# now connect the trajectories by looking at the list of boxes
import csv
with open(filename+'.MassParticles.csv', "rU") as aggCSV: #use to compare values # http://stackoverflow.com/questions/26102302/python-csv-read-error
    reader = csv.reader(aggCSV,delimiter=',')
    masses = list(reader)
aggCSV.close()
masses=sorted(masses, key=lambda sss: sss[0])
masses=masses[-1:] + masses[:-2] # Put the last row into the first, which is the name row
masses=masses[1][1:] # set the masses of the particles on the first frame, use these for the masses of all the particles, the add together when they aggregate
masses = [ float(x) for x in masses ] # change the masses to floats
for iii in xrange(0,len(masses)):
    if(masses[iii]>1):
        masses[iii]=int(masses[iii]+0.5) # set the masses to whole numbers
    else:
        masses[iii]=int(1)

with open(filename+'.BoxLocation.csv', "rU") as aggCSV: #use to compare values # http://stackoverflow.com/questions/26102302/python-csv-read-error
    reader = csv.reader(aggCSV,delimiter=',')
    boxes = list(reader)
aggCSV.close()
#boxes=boxes[-1:] + boxes[:-2] # Put the last row into the first, which is the name row


TList=[]
for ttt in xrange(1,len(boxes)): # loop through the times
    TList.append(float(boxes[ttt][0])) # a list of the times

BoxLocationList=[]
for parA in xrange(0,len(boxes)-1): # initialize the array to hold the trajectory
    BoxLocationList.append([None]*((len(boxes[1])-2)/9)) # BoxLocationList[time][ParticleID]=4 corners, the Particle ID is defined by the first frame's particles

for iii in xrange(2,len(boxes[1]),9): # first time of the trajectory list
    BoxLocationList[0][(iii-2)/9]=[[float(boxes[1][iii]),float(boxes[1][iii+1])],[float(boxes[1][iii+2]),float(boxes[1][iii+3])],[float(boxes[1][iii+4]),float(boxes[1][iii+5])],[float(boxes[1][iii+6]),float(boxes[1][iii+7])]]
    open(filename+'.'+str((iii-2)/9)+'.ParticleLocation.csv','w').write(str(boxes[1][iii])+','+str(boxes[1][iii+1])+','+str(boxes[1][iii+2])+','+str(boxes[1][iii+3])+','+str(boxes[1][iii+4])+','+str(boxes[1][iii+5])+','+str(boxes[1][iii+6])+','+str(boxes[1][iii+7]))

if CalculateTrajectories == 0: # end the program if you don't need trajectories
    os.system("rm *.simple.*.png")
    os.system("rm *.ParticleLocation.csv")
    calculationsf.write('Diffusion Coefficient from Stokes-Einstein equation in Micrometers^2 Per second'+','+str(DSE)+'\n') # http://en.wikipedia.org/wiki/Einstein_relation_(kinetic_theory)#Stokes-Einstein_equation # https://en.wikipedia.org/wiki/Viscosity#Water # (1.380*10^-23)*(J/K)*298*K/(6*pi*0.00089*Pa*s) in microns^3/s
    calculationsf.write('The characteristic Brownian time in seconds'+','+str(tBrown)+'\n')
    calculationsf.write('Rotational Diffusion Time (s)'+','+str(TauR)+'\n') # http://www.wolframalpha.com/input/?i=8*pi*%28mPa*s%29*%281*microns%29%5E3%2F%284.11*10%5E-21*J%29 # http://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.088304 #https://en.wikipedia.org/wiki/Rotational_diffusion#Basic_equations_of_rotational_diffusion
    calculationsf.write('Number of particles: '+','+str(nParticles)+'\n')
    calculationsf.write('Number of chains: '+','+str(sum(TZeroList))+'\n')
    #if(len(TList)>3 and TList[1]>0 and TList[-1]>0): calculationsf.write('Slope of Log-Log Length vs time: '+','+str(math.log(LengthList[-1]/LengthList[1])/math.log(TList[-1]/TList[1]))+'\n')
    SurfaceConc=nParticles/(heightV*widthV*MicronPerPix**2)
    calculationsf.write('Height(pix), Width(pix): '+','+str(heightV)+','+str(widthV)+'\n')
    calculationsf.write('Height(micron), Width(micron): '+','+str(heightV*MicronPerPix)+','+str(widthV*MicronPerPix)+'\n')
    calculationsf.write('Height(radii), Width(radii): '+','+str(heightV*MicronPerPix/ParRMicrons)+','+str(widthV*MicronPerPix/ParRMicrons)+'\n')
    calculationsf.write('Frame Area in Radii: '+','+str(FrameAreaRadii)+'\n')
    calculationsf.write('Surface Concentration (particle/micron^2): '+','+str(SurfaceConc)+'\n')
    calculationsf.write('Surface Concentration (chains/micron^2): '+','+str(sum(TZeroList)/FrameAreaMicrons)+'\n')
    calculationsf.write('Surface Concentration (particle/radius^2): '+','+str(nParticles/FrameAreaRadii)+'\n')
    calculationsf.write('Surface Concentration (chains/radius^2): '+','+str(sum(TZeroList)/FrameAreaRadii)+'\n')
    calculationsf.write('Initial Fraction of Singlets: '+','+str(TZeroList[0]/nParticles)+'\n')
    calculationsf.write('Surface Area Fraction: '+','+str(SurfaceAreaFraction)+'\n')
    calculationsf.write('Volume Concentration (particle/micron^3): '+','+str(nParticles/(150*heightV*widthV*MicronPerPix**2))+'\n')
    calculationsf.write('Time Elapsed(s): '+','+str(frameno*1.0/fps)+'\n')
    calculationsf.write('Frames per Second: '+','+str(fps)+'\n')
    calculationsf.write('Total Fames '+','+str(frameno)+'\n')
    calculationsf.write('Time when all particles were singlets (seconds)'+','+str(tMinus)+'\n')
    #calculationsf.write('Average Distance Between Nearest Neighbors (microns and frame 1)'+','+str(AverageDis*MicronPerPix)+'\n')
    print("Area Pixels: ",FrameAreaPixels," , Area Microns: ",FrameAreaMicrons," , Area Radii: ",FrameAreaRadii)
    calculationsf.write('Area Pixels'+','+str(FrameAreaPixels)+'\n')
    calculationsf.write('Area Microns'+','+str(FrameAreaMicrons)+'\n')
    calculationsf.write('Area Radii'+','+str(FrameAreaRadii)+'\n')
    calculationsf.write('K11(r^2/par*t)'+','+str(newratesAbs[0])+'\n')
    calculationsf.write('K12'+','+str(newratesAbs[1])+'\n')
    calculationsf.write('K13'+','+str(newratesAbs[2])+'\n')
    calculationsf.write('K22'+','+str(newratesAbs[3])+'\n')
    calculationsf.write('K23'+','+str(newratesAbs[4])+'\n')
    calculationsf.write('K33'+','+str(newratesAbs[5])+'\n')
    calculationsf.write('Viscosity[kg/(microns*s)]'+','+str(Viscosity)+'\n')
    calculationsf.write('Vacuum_permeability[kg*microns*s^-2*A^-2]'+','+str(Vacuum_permeability)+'\n')
    calculationsf.write('Fivemu0__4pisq_mu_r[microns/(s*A^2)]'+','+str(Fivemu0__4pisq_mu_r)+'\n')
    calculationsf.close() # close the calculations file
    quit()

#print("all defined variables: ", dir())
#for name in vars().keys():
#  print(name)
#for value in vars().values():
#  print(value)

#for var, obj in locals().items():
#    print var, sys.getsizeof(obj)

def TrajectoryMatchB(parA,ttt,NewLoc,cornerDisLast,cornerDis):
    for parB in xrange(2,len(boxes[ttt]),9): # loop through the particles in the current time step
        for CornerA in xrange(0,4): # the 4 corners from the -1 time step particle
            for CornerB in xrange(0,4): # the 4 corners from the current time step particle
                cornerDis=math.sqrt((BoxLocationList[ttt-1][parA][CornerA][0]-float(boxes[ttt][parB+2*CornerB]))**2+(BoxLocationList[ttt-1][parA][CornerA][1]-float(boxes[ttt][parB+1+2*CornerB]))**2)
                if(cornerDis<cornerDisLast and cornerDis<DistanceBetweenMovesInitial): # if the corner has a match, break the corner loops
                    NewLoc=[[float(boxes[ttt][parB]),float(boxes[ttt][parB+1])],[float(boxes[ttt][parB+2]),float(boxes[ttt][parB+3])],[float(boxes[ttt][parB+4]),float(boxes[ttt][parB+5])],[float(boxes[ttt][parB+6]),float(boxes[ttt][parB+7])]]
                    cornerDisLast=cornerDis

    BoxLocationList[ttt][parA]=NewLoc
    open(filename+'.'+str(parA)+'.ParticleLocation.csv','a').write('\n'+str(NewLoc[0][0])+','+str(NewLoc[0][1])+','+str(NewLoc[1][0])+','+str(NewLoc[1][1])+','+str(NewLoc[2][0])+','+str(NewLoc[2][1])+','+str(NewLoc[3][0])+','+str(NewLoc[3][1])) #open a text file to write box locations in a file for each particle



def TrajectoryMatch(parA):
    #print("par: "+str(parA)+" out of "+str(len(BoxLocationList[0]))+" , file "+filename)
    for ttt in xrange(1,len(BoxLocationList)): # loop through the times
        NewLoc=BoxLocationList[ttt-1][parA]
        cornerDisLast=2000*ParRPixels # used to find the shortest distance between two contours
        cornerDis=cornerDisLast*10 # use as an input for the loop function
        TrajectoryMatchB(parA,ttt,NewLoc,cornerDisLast,cornerDis)


if(RunInParallel): # if 0 run serial, if 1 run parallel
    #num_cores=2
    print("num_cores",num_cores)
    inputs = range(0, len(BoxLocationList[0]))  # the range of frames
    Parallel(n_jobs=num_cores)(delayed(TrajectoryMatch)(parA) for parA in inputs)
else:
    for parA in xrange(0,len(BoxLocationList[0])): # loop through the particles
        TrajectoryMatch(parA)


BoxLocationListB=[]*(len(BoxLocationList))
for parA in xrange(0,(len(boxes[1])-2)/9):
    with open(filename+'.'+str(parA)+'.ParticleLocation.csv', "rU") as parACSV: #use to compare values # http://stackoverflow.com/questions/26102302/python-csv-read-error
        reader = csv.reader(parACSV,delimiter=',')
        groups = list(reader)
    parACSV.close()
    BoxLocationListB.append(groups[:])
BoxLocationListB=zip(*BoxLocationListB) # transpose the list so that it is [time][particle]

BoxLocationListC=[]
for parA in xrange(0,len(boxes)-1): # initialize the array to hold the trajectory
    BoxLocationListC.append([None]*((len(boxes[1])-2)/9))

for ttt in xrange(0,len(BoxLocationListB)): # loop through the times
    for parA in xrange(0,len(BoxLocationListB[0])):
        BoxLocationListC[ttt][parA]=[float(BoxLocationListB[ttt][parA][0]),float(BoxLocationListB[ttt][parA][1])],[float(BoxLocationListB[ttt][parA][2]),float(BoxLocationListB[ttt][parA][3])],[float(BoxLocationListB[ttt][parA][4]),float(BoxLocationListB[ttt][parA][5])],[float(BoxLocationListB[ttt][parA][6]),float(BoxLocationListB[ttt][parA][7])]

BoxLocationListB=BoxLocationListC
BoxLocationList=BoxLocationListC
os.system("rm "+filename+"."+"*.ParticleLocation.csv")





RonalLocationsf= open(filename+'.RonalLocations.csv','w') # locations for Ronal
RonalLocationsf.write("Aggregate,number_of_particles,x-coordinate,y-coordinate,orientation(-1=left or 1=right),Number of particles total = ,"+str(nParticles)+",Area Fraction=,"+str(nParticles*math.pi/FrameAreaRadii)+",Height(y)=,"+str(heightV/ParRPixels)+",Width(x)=,"+str(widthV/ParRPixels)+",")
ParLocf = open(filename+'.trajectory.csv','w') #open a text file to write Centroids locations
ParLocf.write('Time(seconds)')
Centroids=BoxLocationList # List of center locations of the form: Centroids[time][particleNumber][x/y]
Colors=[0]*len(Centroids[0]) # colors for the trajectory lines of the particles
LocationList=[]
for parA in xrange(0,len(Centroids[0])): # loop through the particles
    ParLocf.write(","+str(parA)+"x,"+str(parA)+"y")
    MassOfPar=int(round(masses[parA]))
    if(MassOfPar==0):
        MassOfPar=1
    for parB in xrange(0,MassOfPar): # the location of each particle in the box
        #CentroidTemp = [BoxLocationList[0][parA][0][0],BoxLocationList[0][parA][0][1]]
        CentroidTemp = [(BoxLocationList[0][parA][0][0]+BoxLocationList[0][parA][1][0]+BoxLocationList[0][parA][2][0]+BoxLocationList[0][parA][3][0])/4.0,(BoxLocationList[0][parA][0][1]+BoxLocationList[0][parA][1][1]+BoxLocationList[0][parA][2][1]+BoxLocationList[0][parA][3][1])/4.0]
        LocationList.append([CentroidTemp[0]/ParRPixels+math.sin(30*math.pi/180)*int(2*(parB%2-0.5)),CentroidTemp[1]/ParRPixels-2*parB*math.cos(30*math.pi/180)+MassOfPar*0.5-1 , int(-2*(parB%2-0.5))])
        #if(len(LocationList)>1): print("LocationList[-1]",LocationList[-1],"distance",math.sqrt((LocationList[-1][0]-LocationList[-2][0])**2+(LocationList[-1][1]-LocationList[-2][1])**2),"parA",parA,"parB",parB)
        RonalLocationsf.write("\n"+str(parA)+","+str(int(round(masses[parA])))+","+str(LocationList[-1][0])+","+str(LocationList[-1][1])+ "," +str(int(-2*(parB%2-0.5))) )
RonalLocationsf.close()

RonalLocationsBf= open(filename+'.RonalLocationsB.csv','w') # locations for Ronal
Duplicatesf = open(filename+'.Duplicates.csv','w')
LammpsLocationsf= open('LammpsLocations.csv','w') # locations for Lammps
LammpsLocationsBf= open(filename+'.LammpsLocationsB.csv','w') # locations for Lammps stored for this video
RonalLocationsBf.write("Aggregate,number_of_particles,x-coordinate,y-coordinate,orientation(-1=left or 1=right),Number of particles total = ,"+str(nParticles)+",Area Fraction=,"+str(nParticles*math.pi/FrameAreaRadii)+",Height(y)=,"+str(heightV/ParRPixels)+",Width(x)=,"+str(widthV/ParRPixels)+",")
Duplicatesf.write("Aggregate,number_of_particles,x-coordinate,y-coordinate,orientation(-1=left or 1=right),Number of particles total = ,"+str(nParticles)+",Area Fraction=,"+str(nParticles*math.pi/FrameAreaRadii)+",Height(y)=,"+str(heightV/ParRPixels)+",Width(x)=,"+str(widthV/ParRPixels)+",")
LammpsLocationsf.write("delete_atoms        group all")
LammpsLocationsBf.write("delete_atoms        group all")
frame = cv2.imread(filename+'.simple.'+(str(int(0)).zfill(6))+'.png',0) # print the particle locations on the frame
frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
frameB = frame

b_channel, g_channel, r_channel = cv2.split(frameB)
alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 0 #creating a dummy alpha channel image.
alpha_channel = alpha_channel.astype(np.uint8) #https://stackoverflow.com/questions/32290096/python-opencv-add-alpha-channel-to-rgb-image
frameB = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
frameC = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

cv2.subtract(frame,frame,frameB)
for parA in range(0,len(LocationList)):
    Duplicate=0
    for parB in range(parA+1,len(LocationList)):
        if((LocationList[parA][0]-LocationList[parB][0])**2+(LocationList[parA][1]-LocationList[parB][1])**2<(1.98)**2 and parA != parB):
            Duplicate=1
            print("duplicate!!!!",parA,parB,LocationList[parA],LocationList[parB],(ParRPixels*0.99)**2,(LocationList[parA][0]-LocationList[parB][0])**2+(LocationList[parA][1]-LocationList[parB][1])**2)
    if(Duplicate==0):
        #print(LocationList[parA])
        #print(masses[parA])
        RonalLocationsBf.write("\n"+str(parA)+","+str(int(round(1)))+","+str(LocationList[parA][0])+","+str(LocationList[parA][1])+","+str(LocationList[parA][2])+",")
        LammpsLocationsBf.write("\ncreate_atoms        2 single "+str(LocationList[parA][0])+" "+str(LocationList[parA][1])+" 0")
        LammpsLocationsf.write("\ncreate_atoms        2 single "+str(LocationList[parA][0])+" "+str(LocationList[parA][1])+" 0")
        cv2.circle(frame,(int(LocationList[parA][0]*ParRPixels),int(LocationList[parA][1]*ParRPixels)), int(ParRPixels), (0,0,255), -1)
        cv2.circle(frameB,(int(LocationList[parA][0]*ParRPixels),int(LocationList[parA][1]*ParRPixels)), int(ParRPixels), (0,0,255,255), -1)
        #cv2.putText(frame,str(parA), (int(LocationList[parA][0]*ParRPixels),int(LocationList[parA][1]*ParRPixels)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    else:
        pass
        Duplicatesf.write("\n"+str(parA)+","+str(int(round(1)))+","+str(LocationList[parA][0])+","+str(LocationList[parA][1])+","+str(LocationList[parA][2])+",")
        cv2.circle(frameC,(int(LocationList[parA][0]*ParRPixels),int(LocationList[parA][1]*ParRPixels)), int(ParRPixels), (0,0,255,255), -1)
        cv2.putText(frame,str(parA), (int(LocationList[parA][0]*ParRPixels),int(LocationList[parA][1]*ParRPixels)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,255,255))
        cv2.putText(frameB,str(parA), (int(LocationList[parA][0]*ParRPixels),int(LocationList[parA][1]*ParRPixels)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,255,255))
        cv2.putText(frameC,str(parA), (int(LocationList[parA][0]*ParRPixels),int(LocationList[parA][1]*ParRPixels)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,255,255))
        #print("duplicate!!!! = ",Duplicate,parA,LocationList[parA])
cv2.imwrite(filename+'.RonalLocations.png',frame)
cv2.imwrite(filename+'.RonalLocationsB.png',frameB)
cv2.imwrite(filename+'.Duplicates.png',frameC)
RonalLocationsBf.close()
Duplicatesf.close()
LammpsLocationsf.close()
LammpsLocationsBf.close()

for ttt in xrange(0,len(Centroids)): # loop through the times
    ParLocf.write('\n'+str(TList[ttt]))
    for parA in xrange(0,len(Centroids[ttt])): # loop through the particles
        Colors[parA]=[random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        Centroids[ttt][parA]=[int(0.25*(Centroids[ttt][parA][0][0]+Centroids[ttt][parA][1][0]+Centroids[ttt][parA][2][0]+Centroids[ttt][parA][3][0])),int(0.25*(Centroids[ttt][parA][0][1]+Centroids[ttt][parA][1][1]+Centroids[ttt][parA][2][1]+Centroids[ttt][parA][3][1]))]
        ParLocf.write(","+str(Centroids[ttt][parA][0])+","+str(Centroids[ttt][parA][1]))

# Check the nearest neighbor for the different types "median interparticle distance"
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
print("Collecting Nearest-Neighbor Data")
NearestNeighborOverallf = open(filename+'.NearestNeighborOverall.csv','w')
NearestNeighborOverallf.write("Overall(microns),x(microns),y(microns),Force(Relative),Force(No-Angle),Angle(radians)\n")
NearestNeighbort0f = open(filename+'.NearestNeighbort0.csv','w')
NearestNeighbort0f.write("Overall(microns)\n")
NearestNeighborSingletf = open(filename+'.NearestNeighborSingletSinglet.csv','w')
NearestNeighborSingletDoubletf = open(filename+'.NearestNeighborSingletDoublet.csv','w')
NearestNeighborSingletTripletf = open(filename+'.NearestNeighborSingletTriplet.csv','w')
for ttt in xrange(0,len(Centroids)): # loop through the times
    NearestNeighborsSinglet=[]
    NearestNeighborsDoublet=[]
    NearestNeighborsTriplet=[]
    NearestNeighborsSingletX=[]
    NearestNeighborsDoubletX=[]
    NearestNeighborsSingletY=[]
    NearestNeighborsDoubletY=[]
    frame = cv2.imread(filename+'.simple.'+(str(int(ttt*FrameSampleRate)).zfill(6))+'.png',0)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for parA in xrange(0,len(Centroids[ttt])): # loop through the particles
        SquareDistanceSinglet=FrameAreaPixels # the squared distance for the minimum nearest neighbor between singlets and another singlet
        SquareDistanceDoublet=FrameAreaPixels # the squared distance for the minimum nearest neighbor between doublets and the nearest singlet
        SquareDistanceTriplet=FrameAreaPixels # the squared distance for the minimum nearest neighbor between triplets and the nearest singlet
        SquareDistanceSingletX=FrameAreaPixels
        SquareDistanceDoubletX=FrameAreaPixels
        SquareDistanceSingletY=FrameAreaPixels
        SquareDistanceDoubletY=FrameAreaPixels
        for parB in xrange(0,len(Centroids[ttt])): # loop through the possible nearest neighbor
            #print("Centroids[ttt][parA]",Centroids[ttt][parA])
            SquareDistanceSingletTemp=(Centroids[ttt][parA][0]-Centroids[ttt][parB][0])**2+(Centroids[ttt][parA][1]-Centroids[ttt][parB][1])**2
            SquareDistanceDoubletTemp=SquareDistanceSingletTemp
            SquareDistanceSingletXTemp=(Centroids[ttt][parA][0]-Centroids[ttt][parB][0])
            SquareDistanceDoubletXTemp=SquareDistanceSingletXTemp
            SquareDistanceSingletYTemp=(Centroids[ttt][parA][1]-Centroids[ttt][parB][1])
            SquareDistanceDoubletYTemp=SquareDistanceSingletYTemp
            if(SquareDistanceSingletTemp>0 and SquareDistanceSingletTemp<SquareDistanceSinglet and masses[parA]+masses[parB]==2):
                SquareDistanceSinglet=SquareDistanceSingletTemp
                SquareDistanceSingletX=SquareDistanceSingletXTemp
                SquareDistanceSingletY=SquareDistanceSingletYTemp
                ParA_1_1_Temp=parA
                ParB_1_1_Temp=parB
            elif(SquareDistanceSingletTemp>0 and SquareDistanceDoubletTemp<SquareDistanceDoublet and masses[parA]+masses[parB]==3):
                SquareDistanceDoublet=SquareDistanceDoubletTemp
                SquareDistanceDoubletX=SquareDistanceDoubletXTemp
                SquareDistanceDoubletY=SquareDistanceDoubletYTemp
                ParA_1_2_Temp=parA
                ParB_1_2_Temp=parB
            elif(SquareDistanceSingletTemp>0 and SquareDistanceSingletTemp<SquareDistanceTriplet and ((masses[parA]==1 and masses[parB]==3) or (masses[parA]==3 and masses[parB]==1))):
                SquareDistanceTriplet=SquareDistanceSingletTemp
                ParA_1_3_Temp=parA
                ParB_1_3_Temp=parB
                #print(SquareDistanceTriplet)
        if(SquareDistanceSinglet<FrameAreaPixels and SquareDistanceSinglet>0):
            DistanceSinglet=math.sqrt(SquareDistanceSinglet)
            NearestNeighborsSinglet.append(DistanceSinglet)
            NearestNeighborsSingletX.append(SquareDistanceSingletX)
            NearestNeighborsSingletY.append(SquareDistanceSingletY)
            if(ttt==0):
                theta_radians = math.atan2(SquareDistanceSingletY, SquareDistanceSingletX)
                ForceSimple=1.0/SquareDistanceSinglet**2
                Force=ForceSimple*(3.0*math.cos(theta_radians)**2-1)
                NearestNeighborOverallf.write(str(math.sqrt((SquareDistanceSingletX*MicronPerPix)**2+(SquareDistanceSingletY*MicronPerPix)**2))+",")
                NearestNeighborOverallf.write(str(SquareDistanceSingletX*MicronPerPix)+","+str(SquareDistanceSingletY*MicronPerPix)+","+str(Force)+","+str(ForceSimple)+","+str(theta_radians)+"\n")
            cv2.line(frame, (Centroids[ttt][ParA_1_1_Temp][0],Centroids[ttt][ParA_1_1_Temp][1]), (Centroids[ttt][ParB_1_1_Temp][0],Centroids[ttt][ParB_1_1_Temp][1]), [255,0,0],1)
            cv2.putText(frame,str(round(DistanceSinglet*MicronPerPix)), (Centroids[ttt][ParA_1_1_Temp][0],Centroids[ttt][ParA_1_1_Temp][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0])
        elif(SquareDistanceDoublet<FrameAreaPixels and SquareDistanceDoublet>0):
            DistanceDoublet=math.sqrt(SquareDistanceDoublet)
            NearestNeighborsDoublet.append(math.sqrt(SquareDistanceDoublet))
            NearestNeighborsDoubletX.append(SquareDistanceDoubletX)
            NearestNeighborsDoubletY.append(SquareDistanceDoubletY)
            cv2.line(frame, (Centroids[ttt][ParA_1_2_Temp][0],Centroids[ttt][ParA_1_2_Temp][1]), (Centroids[ttt][ParB_1_2_Temp][0],Centroids[ttt][ParB_1_2_Temp][1]), [0,255,0],1)
            cv2.putText(frame,str(round(DistanceDoublet*MicronPerPix)), (Centroids[ttt][ParA_1_2_Temp][0],Centroids[ttt][ParA_1_2_Temp][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0])
        elif(SquareDistanceTriplet<FrameAreaPixels and SquareDistanceTriplet>0):
            DistanceTriplet=math.sqrt(SquareDistanceTriplet)
            NearestNeighborsTriplet.append(math.sqrt(SquareDistanceTriplet))
            cv2.line(frame, (Centroids[ttt][ParA_1_3_Temp][0],Centroids[ttt][ParA_1_3_Temp][1]), (Centroids[ttt][ParB_1_3_Temp][0],Centroids[ttt][ParB_1_3_Temp][1]), [0,0,255],1)
            cv2.putText(frame,str(round(DistanceTriplet*MicronPerPix)), (Centroids[ttt][ParA_1_3_Temp][0],Centroids[ttt][ParA_1_3_Temp][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255])
    NearestNeighborsSinglet=np.asarray(NearestNeighborsSinglet)
    NearestNeighborsDoublet=np.asarray(NearestNeighborsDoublet)
    NearestNeighborsTriplet=np.asarray(NearestNeighborsTriplet)
    NearestNeighborsSingletX=np.asarray(NearestNeighborsSingletX)
    NearestNeighborsDoubletX=np.asarray(NearestNeighborsDoubletX)
    NearestNeighborsSingletY=np.asarray(NearestNeighborsSingletY)
    NearestNeighborsDoubletY=np.asarray(NearestNeighborsDoubletY)
    if(ttt==0): # for the first time, defime the histogram ranges
        NearestNeighborsSingletHistogram=np.histogram(NearestNeighborsSinglet,density=True)
        NearestNeighborSingletf.write('Time(seconds),MedianDistance(microns),StandardDeviation(microns),MedianX(microns),MedianY(microns),Skew,kurtosis,,'+str((NearestNeighborsSingletHistogram[1][1:]*MicronPerPix).tolist())[1:-1]+',,par1,par2,par3,...')
        NearestNeighborsDoubletHistogram=np.histogram(NearestNeighborsDoublet,density=True)
        NearestNeighborSingletDoubletf.write('Time(seconds),MedianDistance(microns),StandardDeviation(microns),MedianX(microns),MedianY(microns),StandardDeviation(microns),Skew,kurtosis,,'+str((NearestNeighborsDoubletHistogram[1][1:]*MicronPerPix).tolist())[1:-1]+',,par1,par2,par3,...')
        NearestNeighborsTripletHistogram=np.histogram(NearestNeighborsTriplet,density=True)
        NearestNeighborSingletTripletf.write('Time(seconds),MedianDistance(microns),StandardDeviation(microns),MedianX(microns),MedianY(microns),StandardDeviation(microns),Skew,kurtosis,,'+str((NearestNeighborsTripletHistogram[1][1:]*MicronPerPix).tolist())[1:-1]+',,par1,par2,par3,...')
        MaxSingletPDF = 3*np.amax(NearestNeighborsSingletHistogram[0])
        MaxDoubletSingletPDF = 3*np.amax(NearestNeighborsDoubletHistogram[0])
        #print("MaxSingletPDF,MaxDoubletSingletPDF",MaxSingletPDF,MaxDoubletSingletPDF)

    else: # for all other times, use the first time's histogram ranges
        NearestNeighborsSingletHistogram=np.histogram(NearestNeighborsSinglet,bins=NearestNeighborsSingletHistogram[1],density=True)
        NearestNeighborsDoubletHistogram=np.histogram(NearestNeighborsDoublet,bins=NearestNeighborsDoubletHistogram[1],density=True)
        NearestNeighborsTripletHistogram=np.histogram(NearestNeighborsTriplet,bins=NearestNeighborsTripletHistogram[1],density=True)
        for NearNeigh in xrange(0,len(NearestNeighborsSinglet)):
            pass
    NearestNeighborSingletf.write('\n'+str(round(ttt*1.0*FrameSampleRate/fps,1))+','+str(np.median(NearestNeighborsSinglet)*MicronPerPix)+','+str(np.std(NearestNeighborsSinglet)*MicronPerPix)+','+str(np.median(NearestNeighborsSingletX)*MicronPerPix)+','+str(np.median(NearestNeighborsSingletY)*MicronPerPix)+','+str(skew(NearestNeighborsSinglet)*MicronPerPix)+','+str(kurtosis(NearestNeighborsDoublet)*MicronPerPix)+',,'+str(NearestNeighborsSingletHistogram[0].tolist())[1:-1]+',,'+str(np.sort(NearestNeighborsSinglet).tolist())[1:-1])
    NearestNeighborSingletDoubletf.write('\n'+str(round(ttt*1.0*FrameSampleRate/fps,1))+','+str(np.median(NearestNeighborsDoublet)*MicronPerPix)+','+str(np.std(NearestNeighborsDoublet)*MicronPerPix)+','+str(np.median(NearestNeighborsDoubletX)*MicronPerPix)+','+str(np.median(NearestNeighborsDoubletY)*MicronPerPix)+','+str(skew(NearestNeighborsDoublet)*MicronPerPix)+','+str(kurtosis(NearestNeighborsDoublet)*MicronPerPix)+',,'+str(NearestNeighborsDoubletHistogram[0].tolist())[1:-1]+',,'+str(np.sort(NearestNeighborsDoublet).tolist())[1:-1])
    NearestNeighborSingletTripletf.write('\n'+str(round(ttt*1.0*FrameSampleRate/fps,1))+','+str(np.median(NearestNeighborsTriplet)*MicronPerPix)+','+str(np.std(NearestNeighborsTriplet)*MicronPerPix)+','+str(skew(NearestNeighborsTriplet)*MicronPerPix)+','+str(kurtosis(NearestNeighborsTriplet)*MicronPerPix)+',,'+str(NearestNeighborsTripletHistogram[0].tolist())[1:-1]+',,'+str(np.sort(NearestNeighborsTriplet).tolist())[1:-1])
    #print("NearestNeighborsSingletHistogram")
    #print(NearestNeighborsSingletHistogram[0])
    #print(NearestNeighborsSingletHistogram[1])
    #print("NearestNeighborsDoubletHistogram")
    #print(NearestNeighborsDoubletHistogram[0])
    #print(NearestNeighborsDoubletHistogram[1])
    cv2.imwrite(filename+'.Nearest_Neighbor_Image.'+(str(int(ttt*FrameSampleRate)).zfill(6))+'.png',frame)

    plt.figure()
    plt.title('Singlet-Singlet Nearest Neighbor, t='+str(round(ttt*1.0*FrameSampleRate/fps))+'(s)')
    n, bins, patches = plt.hist(NearestNeighborsSinglet*MicronPerPix, normed=1)
    plt.ylim(0, MaxSingletPDF)
    plt.xlim(0, NearestNeighborsSingletHistogram[1][-1]*MicronPerPix)
    plt.axvline(x=np.median(NearestNeighborsSinglet)*MicronPerPix, color='r', linestyle='--')
    plt.savefig(filename+'.NearestNeighborSingletSinglet.'+str(ttt).zfill(6)+'.png')
    plt.figure()
    plt.title('Doublet-Singlet Nearest Neighbor, t='+str(round(ttt*1.0*FrameSampleRate/fps))+'(s)')
    n, bins, patches = plt.hist(NearestNeighborsDoublet*MicronPerPix, normed=1)
    plt.ylim(0, MaxDoubletSingletPDF)
    plt.xlim(0, NearestNeighborsDoubletHistogram[1][-1]*MicronPerPix)
    plt.axvline(x=np.median(NearestNeighborsDoublet)*MicronPerPix, color='r', linestyle='--')
    plt.savefig(filename+'.NearestNeighborSingletDoublet.'+str(ttt).zfill(6)+'.png')
    plt.close("all")

NearestNeighborOverallf.close()
NearestNeighbort0f.close()
NearestNeighborSingletf.close()
NearestNeighborSingletDoubletf.close()
NearestNeighborSingletTripletf.close()
os.system(ffmpegCommand+".Nearest_Neighbor_Image.*.png' -y "+filename+".Nearest_Neighbor_Image.gif")
os.system("rm "+filename+".Nearest_Neighbor_Image.*.png")
os.system(ffmpegCommand+".NearestNeighborSingletSinglet.*.png' -y "+filename+".NearestNeighborSingletSinglet.avi")
os.system("ffmpeg -i "+filename+".NearestNeighborSingletSinglet.avi -y -filter:v 'setpts=10.0*PTS' "+filename+".NearestNeighborSingletSingletB.avi")
os.system("ffmpeg -i "+filename+".NearestNeighborSingletSingletB.avi -y "+filename+".NearestNeighborSingletSinglet.gif")
os.system("rm "+filename+".NearestNeighborSingletSingletB.avi")

os.system("rm "+filename+".NearestNeighborSingletSinglet.*.png")
os.system(ffmpegCommand+".NearestNeighborSingletDoublet.*.png' -y "+filename+".NearestNeighborSingletDoublet.avi")
os.system("ffmpeg -i "+filename+".NearestNeighborSingletDoublet.avi -y -filter:v 'setpts=10.0*PTS' "+filename+".NearestNeighborSingletDoubletB.avi")
os.system("ffmpeg -i "+filename+".NearestNeighborSingletDoubletB.avi -y "+filename+".NearestNeighborSingletDoublet.gif")
os.system("rm "+filename+".NearestNeighborSingletDoubletB.avi")
os.system("rm "+filename+".NearestNeighborSingletDoublet.*.png")

print("Put together groups of particles")
PartGroups=range(0, len(Centroids[-1])) # the indexes of the particles
checkA=0
while (checkA<len(PartGroups)): # the current group
    PartGroups[checkA]=[PartGroups[checkA]] # turn the particle in the list to a list of particles
    checkB=0 # use this check to loop through the particles in the current list
    while (checkB<len(PartGroups[checkA])): # the particles in the current group
        checkC=checkA+1 # use this to loop through all of the other particles and check for matches
        while (checkC<len(PartGroups)):
            if(Centroids[-1][PartGroups[checkA][checkB]]==Centroids[-1][PartGroups[checkC]]): # if the groups are the same at the end, treat them as a group
                PartGroups[checkA].append(PartGroups[checkC])
                del PartGroups[checkC]
            checkC+=1
        checkB+=1
    checkA+=1

# Now loop through the times backwards until there is a collision, or a difference between the positions, from position to position, use for Diffusion constant of the aggregate, for position to collision, do the same, for collision to anything else, do deterministic fit for a certain time back
DsList=[] # The list of diffusion coefficients, calculated by the MSD of particles of various weights
C5s=[] # the magnetic dipole constant C5(mass.i) = ((x2-x1)+(y2-y1)**2)^2.5/(t2-t1)
CollisionVsTime=[] # collisions of each type for each time. Call to a collision as CollisionVsTime[time index][size]
FormationsVsTime=[]
AggregateFromCollisions=[]
CollisionFullCount=[] # Collisions of the CollisionFullCount[time][particleA][particleB(smaller one)]
for iii in xrange(0,100):
    DsList.append([])
    C5s.append([])
    for jjj in xrange(0,100):
        C5s[-1].append([])


for ttt in xrange(0,len(Centroids)): # CollisionVsTime[time_index][cluster consumed]
    CollisionVsTime.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    AggregateFromCollisions.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    FormationsVsTime.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    CollisionFullCount.append([])
    for ppp in xrange(0,100):
        CollisionFullCount[-1].append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])


KeyTList=[] # KeyTList[group][group_id_of_particle][0=Timeid,1=particleID,3=location]
for ggg in xrange(0,len(PartGroups)): # loop through the groups
    for ppp in xrange(0,len(PartGroups[ggg])): # loop through the particles
        KeyT=[[len(Centroids)-2,PartGroups[ggg][ppp],Centroids[-2][PartGroups[ggg][ppp]]]] # Key Times, the final time (needs one offset for collisions), the collisions, and the initial time
        for pppB in xrange(0,len(PartGroups[ggg])): # loop through the particles that are partners in the group
            for ttt in xrange(len(Centroids)-1,-1,-1): #loop backwards through the times, compare a time with the time right before
                if(Centroids[ttt][PartGroups[ggg][ppp]] != Centroids[ttt][PartGroups[ggg][pppB]]):
                    KeyT.append([ttt,PartGroups[ggg][pppB],Centroids[ttt][PartGroups[ggg][ppp]]]) # Append the frame right before the collision time of each partner
                    break
        if(KeyT[-1][0] != 0.0):
            KeyT.append([0,PartGroups[ggg][ppp],Centroids[0][PartGroups[ggg][ppp]]]) # if the last collision is not at t-zero, make a time for t-zero
        KeyTList.append(KeyT) # note all the key times

    temp_Group = [PartGroups[ggg]] # list of a list that will break up as the particles break up, accounting of groups in this group
    for pppC in xrange(1,len(KeyT)): #loop through all the key times for a group, count all events, this is done reverse chronologically
        for pppD in xrange(0,len(temp_Group)): # loop theough each groups in this group, account for each breakup
            for pppE in xrange(0,len(temp_Group[pppD])-1): # loop through each particle in the subgroup, besides the last, this is for pairs
                pppF=pppE+1
                NewGroup=[] # the new group of particles where the broken particles go
                while (pppF<len(temp_Group[pppD])): # loop through each particle-pair in the subgroup, needs to be dynamic for
                    if(Centroids[KeyT[pppC][0]][temp_Group[pppD][pppE]] != Centroids[KeyT[pppC][0]][temp_Group[pppD][pppF]]):
                        #print("time,breakup",KeyT[pppC][0],temp_Group[pppD][pppF])
                        NewGroup.append(temp_Group[pppD][pppF]) # make a new group with the broken up particles
                        del temp_Group[pppD][pppF]
                        temp_Group.append(NewGroup)#add the new group as a subgroup to the temp group
                    pppF+=1
                if(len(NewGroup)>0): # if there is a breakup, then account what happened in the collision list
                    RemainingMass=0
                    NewGroupMass=0
                    for pppG in xrange(0,len(temp_Group[pppD])):#sum the masses in the Remaining Group
                        RemainingMass+=masses[temp_Group[pppD][pppG]]
                    for pppG in xrange(0,len(NewGroup)):#sum the masses in the Remaining Group
                        NewGroupMass+=masses[NewGroup[pppG]]
                    CollisionVsTime[KeyT[pppC][0]][RemainingMass-1]+=1
                    CollisionVsTime[KeyT[pppC][0]][NewGroupMass-1]+=1
                    FormationsVsTime[KeyT[pppC][0]][NewGroupMass+RemainingMass-1]+=1
                    CollisionFullCount[KeyT[pppC][0]][RemainingMass-1][NewGroupMass-1]+=1
                    if(NewGroupMass != RemainingMass):
                        CollisionFullCount[KeyT[pppC][0]][NewGroupMass-1][RemainingMass-1]+=1 # assign the increment to [i][j] and its reflection [j][i]



for ggg in xrange(0,len(TZeroList)):
    for ttt in xrange(0,len(Centroids)):
        AggregateFromCollisions[ttt][ggg]+=TZeroList[ggg]

CollisionVsTimef = open(filename+'.CollisionVsTime.csv','w') #open a text file to write collisions
CumCollisionVsTimef = open(filename+'.CumCollisionVsTime.csv','w') #open a text file to write cumulative collisions
CumCollision_1nf = open(filename+'.CumCollision_1n.csv','w') #open a text file to write cumulative collisions to each kind
CumCollision_1nf.write("Time(ND),1-1,1-2,1-3,1-4,1-5,1-6,1-7,1-8,1-9,1-10,")
CumCollision_2nf = open(filename+'.CumCollision_2n.csv','w') #open a text file to write cumulative collisions to each kind
CumCollision_2nf.write("Time(ND),2-1,2-2,2-3,2-4,2-5,2-6,2-7,2-8,2-9,2-10,")
CumCollision_3nf = open(filename+'.CumCollision_3n.csv','w') #open a text file to write cumulative collisions to each kind
CumCollision_3nf.write("Time(ND),3-1,3-2,3-3,3-4,3-5,3-6,3-7,3-8,3-9,3-10,")
CumCollision_4nf = open(filename+'.CumCollision_4n.csv','w') #open a text file to write cumulative collisions to each kind
CumCollision_4nf.write("Time(ND),4-1,4-2,4-3,4-4,4-5,4-6,4-7,4-8,4-9,4-10,")
CumCollisionVsTimef.write("Time(s),Singlets,Doublets,Triplets,Quadruplets,Pentuplets,6-let,7-let,8-let,9-let,10-let,")
AggregateFromCollisionsf = open(filename+'.AggregateFromCollisions.csv','w') #open a text file to write collisions
AggregateFromCollisionsf.write("Time(ND),Singlets,Doublets,Triplets,Quadruplets,Pentuplets,6-let,7-let,8-let,9-let,10-let,")
for ttt in xrange(1,len(Centroids)):
    CollisionVsTimef.write(str(CollisionVsTime[ttt])[1:-1]+"\n")
    CumCollisionVsTimef.write("\n"+str(TList[ttt])+",")
    AggregateFromCollisionsf.write("\n"+str(TList[ttt]/tBrown)+",")
    CumCollision_1nf.write("\n"+str(TList[ttt]/tBrown)+",")
    CumCollision_2nf.write("\n"+str(TList[ttt]/tBrown)+",")
    CumCollision_3nf.write("\n"+str(TList[ttt]/tBrown)+",")
    CumCollision_4nf.write("\n"+str(TList[ttt]/tBrown)+",")
    for ggg in xrange(0,len(CollisionVsTime[ttt])): # loop through the particles
        CollisionVsTime[ttt][ggg]+=CollisionVsTime[ttt-1][ggg]
        FormationsVsTime[ttt][ggg]+=FormationsVsTime[ttt-1][ggg]
        CumCollisionVsTimef.write(str(CollisionVsTime[ttt][ggg])+",")
        AggregateFromCollisions[ttt][ggg]-=CollisionVsTime[ttt-1][ggg]
        AggregateFromCollisions[ttt][ggg]+=FormationsVsTime[ttt-1][ggg]
        AggregateFromCollisionsf.write(str(AggregateFromCollisions[ttt][ggg])+",")
        CollisionFullCount[ttt][0][ggg]+=CollisionFullCount[ttt-1][0][ggg]
        CollisionFullCount[ttt][1][ggg]+=CollisionFullCount[ttt-1][1][ggg]
        CollisionFullCount[ttt][2][ggg]+=CollisionFullCount[ttt-1][2][ggg]
        CollisionFullCount[ttt][3][ggg]+=CollisionFullCount[ttt-1][3][ggg]
        CumCollision_1nf.write(str(CollisionFullCount[ttt][0][ggg])+",")
        CumCollision_2nf.write(str(CollisionFullCount[ttt][1][ggg])+",")
        CumCollision_3nf.write(str(CollisionFullCount[ttt][2][ggg])+",")
        CumCollision_4nf.write(str(CollisionFullCount[ttt][3][ggg])+",")
CollisionVsTimef.close()
CumCollisionVsTimef.close()
AggregateFromCollisionsf.close()
CumCollision_1nf.close()
CumCollision_2nf.close()
CumCollision_3nf.close()
CumCollision_4nf.close()

#print("AggregateFromCollisions",AggregateFromCollisions)
#raw_input('Press Enter')


MSDParf = open(filename+'.MSDParticles.csv','w') #open a text file to write MSD
MSDParMicronf = open(filename+'.MSDParticles_Microns.csv','w') #open a text file to write MSD in microns
MSDParMicronMassf = open(filename+'.MSDParticles_Microns_Mass.csv','w') #open a text file to write MSD in microns
MSDParMicronCumListf = open(filename+'.MSDParticles_Microns_Cumulative.csv','w') #Cumulative List of chain types MSD
MSDf = open(filename+'.MSD.csv','w') #open a text file to write box locations
MSDParClusterTypesf= open(filename+'.MSDParClusterTypes.csv','w')
MSDParf.write("Time")
MSDParClusterTypesf.write("Time(s),Singlets(microns^2),Doublets(microns^2),Triplets(microns^2),Quadruplets(microns^2),Pentuplets(microns^2)")
MSDParMicronCumListf.write("Time(s),MSD-1(microns^2),MSD-2(microns^2),MSD-3(microns^2),MSD-4(microns^2),MSD-5(microns^2),MSD-6(microns^2),MSD-7(microns^2),MSD-8(microns^2),MSD-9(microns^2),MSD-10(microns^2)")
MSDParMicronf.write("Time")
MSDParMicronMassf.write("Time(s)")
MSDf.write("Time(s),MSD(micron^2),Diffusion_Coefficient(microns^2/s),MSD(pix^2),Median_MSD(pix^2),DisplacementX(pix),DisplacementY(pix),Square_Displacement_Magnitude(pix^2),Effective_Diffusion(microns^2/s)")
massesTot=[] # the total mass at the current time for the aggregate
for parA in xrange(0,len(Centroids[ttt])): # loop through the particles
    MSDParf.write(",Par"+str(parA)) # write the particle number in the header
    MSDParMicronf.write(",Par"+str(parA)) # write the particle number in the header
    MSDParMicronMassf.write(",Mass:"+str(masses[parA])) # write the particle mass in the header
    massesTot.append(masses[parA])
MSDPar = [] # Centroids[time][particleNumber][x/y] # all of the MSD # MSDPar[time][particleNumber][MSD]
Displacement=[]
MSDCumList=[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]] # the cumulative list of Mean Squared Displacement for singlets, doublets, triplets, etc
MSDCumListCount=[0,0,0,0,0,0,0,0,0,0] # the count of the number of chains of this type for the current MSD
for ttt in xrange(0,len(Centroids)-1): # loop through the times
    MSDPar.append([])
    Displacement.append([])
    MSDParf.write("\n"+str(TList[ttt])) #Write the time
    MSDParMicronf.write("\n"+str(TList[ttt])) #Write the time
    MSDParMicronMassf.write("\n"+str(TList[ttt])) #Write the time
    MSDParMicronCumListf.write("\n"+str(TList[ttt])) #Write the time
    MSDCumList.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    MSDCumListCount=[0,0,0,0,0,0,0,0,0,0]
    for parA in xrange(0,len(Centroids[ttt])): # loop through the particles
        MeanSquaredDisPar=(Centroids[ttt+1][parA][0]-Centroids[0][parA][0])**2 + (Centroids[ttt+1][parA][1]-Centroids[0][parA][1])**2 # Displacement from time zero
        MeanSquaredDisParDX=(Centroids[ttt+1][parA][0]-Centroids[ttt][parA][0])**2 + (Centroids[ttt+1][parA][1]-Centroids[ttt][parA][1])**2 # Displacement from current time
        MSDPar[ttt].append(MeanSquaredDisPar)
        MSDParf.write(','+str(MeanSquaredDisPar))
        MSDParMicronf.write(','+str(MeanSquaredDisPar*MicronPerPix**2))
        #print("KeyTList[parA]",KeyTList[parA])
        #print("KeyTList[parA][1:-1]",KeyTList[parA][1:-1])
        TempCollisions = [item[0] for item in KeyTList[parA][1:-1]]
        TempMasses=[item[1] for item in KeyTList[parA][1:-1]]
        #print(TempCollisions)
        if(ttt in TempCollisions):
            #MSDParMicronMassf.write(',Aggregation: '+str(MeanSquaredDisParDX*MicronPerPix**2))
            ParB=TempMasses[TempCollisions.index(ttt)] #the particle in the collision
            massesTot[parA]+=masses[ParB]
            MSDParMicronMassf.write(",MassB:"+str(masses[ParB])+"/MassTot:"+str(massesTot[parA]))
        else:
            MSDParMicronMassf.write(','+str(MeanSquaredDisParDX*MicronPerPix**2))
            if(massesTot[parA]<11):
                MSDCumList[-1][massesTot[parA]-1]+=MeanSquaredDisParDX*MicronPerPix**2
                MSDCumListCount[massesTot[parA]-1]+=1
        Displacement[ttt].append([Centroids[ttt+1][parA][0]-Centroids[0][parA][0],Centroids[ttt+1][parA][1]-Centroids[0][parA][1]]) # The displacement in the X and Y direction
    for iii in xrange(0,10): # loop through the MSD lists for singlets, doublets, etc
        if(MSDCumListCount[iii]>0):
            MSDCumList[-1][iii]=MSDCumList[-1][iii]/MSDCumListCount[iii]
        MSDCumList[-1][iii]+=MSDCumList[-2][iii]
    MSDParMicronCumListf.write(","+str(MSDCumList[-1])[1:-1]) # cumulative list of the MSD
    DisplacementXTemp=sum(Displacement[-1][:][0])/len(Displacement[-1][:][0])
    DisplacementYTemp=sum(Displacement[-1][:][1])/len(Displacement[-1][:][1])
    SqrDisplacementMagTemp=DisplacementXTemp**2+DisplacementYTemp**2
    MSDTemp=sum(MSDPar[-1])/len(MSDPar[-1]) # the mean squared displacement over the time step
    MSDMedian=np.median(MSDPar[-1])
    TimeStep=TList[ttt+1]-TList[0] # the time step
    DiffusionTemp=MSDTemp*MicronPerPix**2/(4*TimeStep)
    EffectiveDiffusionTemp=(math.sqrt(MSDTemp)-math.sqrt(SqrDisplacementMagTemp))**2*MicronPerPix**2/(4*TimeStep)
    MSDf.write("\n"+str(TList[ttt])+","+str(MSDTemp*MicronPerPix**2)+","+str(DiffusionTemp)+","+str(MSDTemp)+","+str(MSDMedian)+","+str(DisplacementXTemp)+","+str(DisplacementYTemp)+","+str(SqrDisplacementMagTemp)+","+str(EffectiveDiffusionTemp)) #Write the time
MSDf.close()
MSDParf.close()
MSDParMicronf.close()
MSDParMicronMassf.close()

CentroidsClusterTypes=[[],[],[],[],[],[],[],[],[],[]] # the list of locations of various particle types. This is required as aggregating particles vary in type throught the run
MSDClusterTypes=[[],[],[],[],[],[],[],[],[],[]] # the list of MSD of various particle types. This is required as aggregating particles vary in type throught the run
MSDClusterTypesMasked=[[],[],[],[],[],[],[],[],[],[]] # the list of MSD of various particle types. This is required as aggregating particles vary in type throught the run
for parA in xrange(0,len(Centroids[0])): # loop through the particles
    massesTot.append(masses[parA]) # the mass of a given particle
for parA in xrange(0,len(Centroids[0])): # loop through the particles
    if(massesTot[parA]<len(CentroidsClusterTypes)):
        CentroidsClusterTypes[massesTot[parA]-1].append([Centroids[0][parA]]) #append the first location to the location of cluster types
        for ttt in xrange(1,len(Centroids)-1): # loop through the times
            TempCollisions = [item[0] for item in KeyTList[parA][1:-1]]
            if(ttt in TempCollisions):
                massesTot[parA]+=masses[ParB]
                if(massesTot[parA]<len(CentroidsClusterTypes)):
                    CentroidsClusterTypes[massesTot[parA]-1].append([Centroids[ttt][parA]]) #append the first location to the location of cluster types
            elif(massesTot[parA]<len(CentroidsClusterTypes)): # append the current mass if it is of a particular size
                CentroidsClusterTypes[massesTot[parA]-1][-1].append(Centroids[ttt][parA])

import time
def autocorrFFT(x):
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return res/n #this is the autocorrelation in convention A

def msd_fft(r):
    N=len(r)
    D=np.square(r).sum(axis=1)
    D=np.append(D,0)
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    return S1-2*S2

for iii in xrange(0,len(CentroidsClusterTypes)): # loop through the cluster types
    if(len(CentroidsClusterTypes[iii])>0):
        MaxLength=len(max(CentroidsClusterTypes[iii],key=len)) # the MSD cluster that exists for the longest period of time
        MSDClusterTypesMasked[iii] = np.ma.empty((len(CentroidsClusterTypes[iii]),MaxLength))
        MSDClusterTypesMasked[iii].mask = True
        for jjj in xrange(0,len(CentroidsClusterTypes[iii])): # loop through the individual clusters in a given cluster type
            CentroidsClusterTypes[iii][jjj]=np.array(CentroidsClusterTypes[iii][jjj])
            MSDTempCluster=msd_fft(CentroidsClusterTypes[iii][jjj])
            MSDClusterTypes[iii].append(MSDTempCluster)
            MSDClusterTypesMasked[iii][jjj,:len(MSDTempCluster)] = MSDTempCluster


for ttt in xrange(0,len(Centroids)-1):
    MSDParClusterTypesf.write("\n"+str(TList[ttt])) #Write the time
    for iii in xrange(0,len(CentroidsClusterTypes)): # loop through the cluster types
        if(len(CentroidsClusterTypes[iii])>0):
            if(len(MSDClusterTypesMasked[iii].mean(axis = 0))>ttt):
                MSDParClusterTypesf.write(","+str(MSDClusterTypesMasked[iii].mean(axis = 0)[ttt]))
            else:
                MSDParClusterTypesf.write(",")
        else:
            MSDParClusterTypesf.write(",")



print(MSDClusterTypesMasked[0])
print("completed!")
print("Average MSD",MSDClusterTypesMasked[0].mean(axis = 0))

MSDParClusterTypesf.close()


'''
#Singlet particles that do not aggregate throughout the entirety of the experiment, plot their MSD
with open(filename+'.MSDParticles_Microns_Cumulative.csv', "rU") as aggCSV: #use to compare values # http://stackoverflow.com/questions/26102302/python-csv-read-error
    reader = csv.reader(aggCSV,delimiter=',') 
    groups = list(reader) 
aggCSV.close() 
Timed=np.array([float(row[0]) for row in groups][1:]) #Time 
SingletMSDd=Timed*0 #Average singlet MSD
for iii in xrange(1,len(groups)):
    TempSingletMSDd=np.array([float(row[1]) for row in groups][1:])
    if(TempSingletMSDd.dtype==dtype('float64')):
        SingletMSDd+=TempSingletMSDd
print("Timed",Timed)
print("SingletMSDd",SingletMSDd)
print("groups",groups)
'''

Pairf = open(filename+'.Pairs.csv','w') #open a text file to write box locations
Pair11f = open(filename+'.Pairs1-1.csv','w') #open a text file to write pair locations of Singlet-Singlet Aggregation
Pair12f = open(filename+'.Pairs1-2.csv','w') #open a text file to write pair locations of Singlet-Doublet Aggregation
Pair13f = open(filename+'.Pairs1-3.csv','w') #open a text file to write pair locations of Singlet-Triplet Aggregation
C5sf = open(filename+'.C5s.csv','w') #open a text file to write box locations
C5_11sf = open(filename+'.C5s_11.csv','w') #open a text file to write box locations
C5_12sf = open(filename+'.C5s_12.csv','w') #open a text file to write box locations
C5_13sf = open(filename+'.C5s_13.csv','w') #open a text file to write box locations
C5sf.write("ParticleA,ParticleB,massA,massB,angle,C5,dipole(A*micron^2),time(frame),time(s),Distance(microns)\n")
C5_11sf.write("ParticleA,ParticleB,massA,massB,angle,C5,dipole(A*micron^2),time(frame),time(s),Distance(microns)\n")
C5_12sf.write("ParticleA,ParticleB,massA,massB,angle,C5,dipole(A*micron^2),time(frame),time(s),Distance(microns)\n")
C5_13sf.write("ParticleA,ParticleB,massA,massB,angle,C5,dipole(A*micron^2),time(frame),time(s),Distance(microns)\n")
#C5toMsq=(MicronPerPix**5)*ParRMicrons*0.000000006283 #multiply C5 by this to get m1*m2 dipole^2 [] # http://www.wolframalpha.com/input/?i=(4*pi%5E2*mPa*s*microns)*(microns%5E5%2Fs)%2F(5*(4*pi*10%5E-7)*N%2FA%5E2)+in+microns%5E4*A%5E2
Pairf.write("Time(s)")
Pair11f.write("Time(s)")
Pair12f.write("Time(s)")
Pair13f.write("Time(s)")
PairData=[] # Singlet-Singlet Pair Data
PairDataX=[]
PairDataY=[]
PairData_12=[] # Singlet-Doublet Pair Data
PairData_13=[] # Singlet-Triplet Pair Data

CollisionCount=[]
for ttt in xrange(0,FinalFrame,FrameSampleRate): # make a list of aggregation events for every frame
    CollisionCount.append([])
    for iii in xrange(0,50):
        CollisionCount[-1].append([])
        for jjj in xrange(0,50):
            CollisionCount[-1][-1].append(0)


# sort KeyTList[kkk][pppB] by time code at KeyTList[kkk][pppB][0]
# if KeyTList[kkk][pppB][0] is equal to another group member, do not do the evaluation as no collision occured at that key time
# I think it will be: if(KeyTList[kkk][pppB][2]==KeyTList[kkk][pppB+iii][2])
# with some type of loop
print("Collect and write C5 data for all pair aggregation events")
for kkk in xrange(0,len(KeyTList)): # loop through the groups in the key list
    #if(len(KeyTList)>2): Pairf.write(",") # only print groups that have more than one particle
    KeyTList[kkk].sort(reverse=True)
    #print("KeyTList[kkk]",KeyTList[kkk])
    for pppB in xrange(1,len(KeyTList[kkk])-1): # loop through the particles that are partners in the group
        timeID=KeyTList[kkk][pppB][0]
        parID=KeyTList[kkk][pppB][1]
        timeTemp=KeyTList[kkk][pppB][0]-PeOneSamples
        particleA=KeyTList[kkk][0][1] # index particle to compare to the others
        particleB=KeyTList[kkk][pppB][1] # particle to compare to the index
        angleTemp=0
        C5sTemp=0
        disTemp=0
        xPairDis=Centroids[timeTemp][particleA][0]-Centroids[timeTemp][particleB][0] # the squared-x distance between one particle and its partner
        yPairDis=Centroids[timeTemp][particleA][1]-Centroids[timeTemp][particleB][1] # the squared-y distance between one particle and its partner
        xPairDis=xPairDis*MicronPerPix
        yPairDis=xPairDis*MicronPerPix
        PairDis=math.sqrt(xPairDis**2+yPairDis**2)

        xPairDisSq=(Centroids[timeTemp][particleA][0]-Centroids[timeTemp][particleB][0])**2 # the squared-x distance between one particle and its partner
        yPairDisSq=(Centroids[timeTemp][particleA][1]-Centroids[timeTemp][particleB][1])**2 # the squared-y distance between one particle and its partner
        disTemp=math.sqrt(xPairDisSq+yPairDisSq)*MicronPerPix
        timeDTempSec=PeOneSec

        #print("kkk",kkk,"KeyTList[kkk]",KeyTList[kkk],"pppB,len(KeyTList[kkk])",pppB,len(KeyTList[kkk])-2,"particleA,particleB",particleA,particleB)
        #if(kkk>0 and KeyTList[kkk][0][2]!=KeyTList[kkk-1][0][2]):
        if(disTemp>0):
            CollisionCount[timeID][masses[particleA]-1][masses[particleB]-1]+=1
            #print("kkk",kkk,"KeyTList[kkk]",KeyTList[kkk],"pppB,len(KeyTList[kkk])",pppB,len(KeyTList[kkk])-2,"particleA,particleB",particleA,particleB,"Distance",(xPairDis**2+yPairDis**2))
            if(1): #if there are enough frames for the entire PeOneFrames analysis
                #xPairDis=Centroids[timeTemp][particleA][0]-Centroids[timeTemp][particleB][0] # the squared-x distance between one particle and its partner
                #yPairDis=Centroids[timeTemp][particleA][1]-Centroids[timeTemp][particleB][1] # the squared-y distance between one particle and its partner
                #timeTot=TList[KeyTList[kkk][pppB][0]-PeOneSamples]-TList[KeyTList[kkk][pppB][0]] #time(s) covered over the collision
                if(abs(yPairDis)>0):
                    angleTemp=abs(round(math.atan(xPairDis/yPairDis)*180/math.pi))
                else:
                    angleTemp=0
                #C5sTemp=((Centroids[timeID-1-PeOneFrames][particleA][0]*1.0-Centroids[timeID-1-PeOneFrames][particleB][0])**2+(Centroids[timeID-1-PeOneFrames][particleA][1]*1.0-Centroids[timeID-1-PeOneFrames][particleB][1])**2)**2.5/(-1.0*timeTot)
                C5sTemp=(disTemp)**5/PeOneSec
                C5sf.write(str(particleA)+","+str(particleB)+","+str(masses[particleA])+","+str(masses[particleB])+","+str(angleTemp)+","+str(C5sTemp)+","+str(math.sqrt(abs(C5sTemp/Fivemu0__4pisq_mu_r)))+","+str(timeID)+","+str(PeOneSec)+","+str(disTemp)+"\n")
                C5s[masses[particleA]][masses[particleB]].append(C5sTemp)
                #print("xPairDis,yPairDis",xPairDis,yPairDis)
                #print("particleA",particleA,"particleB",particleB)
                #print("PairDis",PairDis)
                #print("disTemp",disTemp)
                #print("PeOneSec",PeOneSec)
                #print("C5sTemp",C5sTemp)
        else: # calculate C5s for particles with less than PeOneFrames between now and last key time
            timeTemp=KeyTList[kkk][pppB+1][0]+1
            xPairDis=Centroids[timeTemp][particleA][0]-Centroids[timeTemp][particleB][0] # the squared-x distance between one particle and its partner
            yPairDis=Centroids[timeTemp][particleA][1]-Centroids[timeTemp][particleB][1] # the squared-y distance between one particle and its partner
            xPairDis=xPairDis*MicronPerPix
            yPairDis=xPairDis*MicronPerPix
            PairDis=math.sqrt(xPairDis**2+yPairDis**2)

            xPairDisSq=(Centroids[timeTemp][particleA][0]-Centroids[timeTemp][particleB][0])**2 # the squared-x distance between one particle and its partner
            yPairDisSq=(Centroids[timeTemp][particleA][1]-Centroids[timeTemp][particleB][1])**2 # the squared-y distance between one particle and its partner
            disTemp=math.sqrt(xPairDisSq+yPairDisSq)*MicronPerPix

            timeDTempSec=(KeyTList[kkk][pppB][0]-KeyTList[kkk][pppB+1][0])*1.0*FrameSampleRate/fps
            if(disTemp>0):
                if(abs(yPairDis)>0):
                    angleTemp=abs(round(math.atan(xPairDis/yPairDis)*180/math.pi))
                else:
                    angleTemp=0

                #print("timeTemp (frames)",timeTemp)
                #print("particleA",particleA,"particleB",particleB)
                #print("the time traversed from the lask key time, timeDTempSec:",timeDTempSec)
                #print("PairDis",PairDis)
                #print("disTemp",disTemp)
                #print("PeOneSec",timeDTempSec)
                #print("C5sTemp",C5sTemp)
                C5sTemp=(disTemp)**5/timeDTempSec
                C5sf.write(str(particleA)+","+str(particleB)+","+str(masses[particleA])+","+str(masses[particleB])+","+str(angleTemp)+","+str(C5sTemp)+","+str(math.sqrt(abs(C5sTemp/Fivemu0__4pisq_mu_r)))+","+str(timeID)+","+str(timeDTempSec)+","+str(disTemp)+"\n")
                C5s[masses[particleA]][masses[particleB]].append(C5sTemp)
            #print("C5sTemp",C5sTemp)
        Pairf.write(",parA"+str(particleA)+"/parB"+str(particleB)+"/"+str(angleTemp)+"deg/"+str(round(C5sTemp,0))+"_C5")
        if(masses[particleA]+masses[particleB]==2):
            C5_11sf.write(str(particleA)+","+str(particleB)+","+str(masses[particleA])+","+str(masses[particleB])+","+str(angleTemp)+","+str(C5sTemp)+","+str(math.sqrt(abs(C5sTemp/Fivemu0__4pisq_mu_r)))+","+str(timeID)+","+str(timeDTempSec)+","+str(disTemp)+"\n")
            Pair11f.write(",parA"+str(particleA)+"/parB"+str(particleB)+"/"+str(angleTemp)+"deg/"+str(round(C5sTemp,0))+"_C5")
        elif(masses[particleA]+masses[particleB]==3):
            C5_12sf.write(str(particleA)+","+str(particleB)+","+str(masses[particleA])+","+str(masses[particleB])+","+str(angleTemp)+","+str(C5sTemp)+","+str(math.sqrt(abs(C5sTemp/Fivemu0__4pisq_mu_r)))+","+str(timeID)+","+str(timeDTempSec)+","+str(disTemp)+"\n")
            Pair12f.write(",parA"+str(particleA)+"/parB"+str(particleB)+"/"+str(angleTemp)+"deg/"+str(round(C5sTemp,0))+"_C5")
        elif((masses[particleA]==1 and masses[particleB]==3) or (masses[particleA]==3 and masses[particleB]==1)):
            C5_13sf.write(str(particleA)+","+str(particleB)+","+str(masses[particleA])+","+str(masses[particleB])+","+str(angleTemp)+","+str(C5sTemp)+","+str(math.sqrt(abs(C5sTemp/Fivemu0__4pisq_mu_r)))+","+str(timeID)+","+str(timeDTempSec)+","+str(disTemp)+"\n")
            Pair13f.write(",parA"+str(particleA)+"/parB"+str(particleB)+"/"+str(angleTemp)+"deg/"+str(round(C5sTemp,0))+"_C5")
        PairData.append([]) #PairData[pairID][t]
        PairDataX.append([]) #PairDataX[pairID][t]
        PairDataY.append([]) #PairDataY[pairID][t]
        PairData_12.append([])
        PairData_13.append([])
C5sf.close()
C5_11sf.close()
C5_12sf.close()
C5_13sf.close()

Collisionsf = open(filename+'.Collisions.csv','w') #open a text file to write when collisions occur
Collisionsf.write("Time, 1-1, 1-2, 1-3, 1-4, 1-5, 2-2, 2-3")
Collisions_Cumulativef = open(filename+'.Collisions_Cumulative.csv','w') #open a text file to write when collisions occur
Collisions_Cumulativef.write("Time, 1-1, 1-2, 1-3, 1-4, 1-5, 2-2, 2-3")
SingletSingletAggregation=[]
SingletSingletAggregation_CumSum=0
SingletDoubletAggregation=[]
SingletDoubletAggregation_CumSum=0
SingletTripletAggregation=[]
SingletTripletAggregation_CumSum=0
SingletQuadrupletAggregation=[]
SingletQuadrupletAggregation_CumSum=0
for ttt in xrange(1,len(CollisionCount)):
    Collisionsf.write("\n"+str(ttt*1.0*FrameSampleRate/fps))
    Collisions_Cumulativef.write("\n"+str(ttt*1.0*FrameSampleRate/fps))

    SingletSingletAggregation.append(CollisionCount[ttt][0][0]) #1-1
    SingletSingletAggregation_CumSum+=CollisionCount[ttt][0][0] #1-1
    Collisionsf.write(","+str(SingletSingletAggregation[-1])) #1-1
    Collisions_Cumulativef.write(","+str(SingletSingletAggregation_CumSum)) #1-1

    SingletDoubletAggregation.append(CollisionCount[ttt][0][1]+CollisionCount[ttt][1][0]) # 1-2, 2-1
    SingletDoubletAggregation_CumSum+=CollisionCount[ttt][0][1]+CollisionCount[ttt][1][0] # 1-2, 2-1
    Collisionsf.write(","+str(SingletDoubletAggregation[-1])) # 1-2, 2-1
    Collisions_Cumulativef.write(","+str(SingletDoubletAggregation_CumSum)) # 1-2, 2-1

    SingletTripletAggregation.append(CollisionCount[ttt][0][2]+CollisionCount[ttt][2][0])
    SingletTripletAggregation_CumSum+=CollisionCount[ttt][0][2]+CollisionCount[ttt][2][0]
    Collisionsf.write(","+str(SingletTripletAggregation[-1])) # 1-3, 3-1
    Collisions_Cumulativef.write(","+str(SingletTripletAggregation_CumSum)) # 1-3, 3-1

    SingletQuadrupletAggregation.append(CollisionCount[ttt][0][3]+CollisionCount[ttt][3][0])
    SingletQuadrupletAggregation_CumSum+=CollisionCount[ttt][0][3]+CollisionCount[ttt][3][0]
    Collisionsf.write(","+str(SingletQuadrupletAggregation[-1])) # 1-4, 4-1
    Collisions_Cumulativef.write(","+str(SingletQuadrupletAggregation_CumSum))

    Collisionsf.write(","+str((CollisionCount[ttt][0][4]+CollisionCount[ttt][4][0]))) # 1-5, 5-1
    Collisionsf.write(","+str((CollisionCount[ttt][1][1]))) # 2-2
    Collisionsf.write(","+str((CollisionCount[ttt][1][2]+CollisionCount[ttt][2][1]))) # 2-3, 3-2
Collisionsf.close()

Collisions_Cumulativef.close()

if(10<int(math.floor(FinalFrame*1.0/FrameSampleRate))):
    N = 10 # number of samples for a moving average
else:
    N = 2
plt.figure()
plt.plot(TList[:-1],SingletSingletAggregation)
plt.plot(TList[:-N],np.convolve(SingletSingletAggregation, np.ones((N,))/N, mode='valid')) #https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
plt.xlabel('Time(s)')
plt.ylabel('Event Count')
plt.savefig(filename+'.singlet_collisions_average.png') #print the plot

plt.figure()
plt.plot(TList[:-N],np.convolve(SingletSingletAggregation, np.ones((N,))/N, mode='valid')*1000000/FrameAreaRadii) #https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
plt.xlabel('Time(s)')
plt.ylabel('10^6 Event Count/Area')
plt.savefig(filename+'.singlet_collisions_average_concentration.png') #print the plot

plt.figure()
plt.plot(TList[:-1],np.cumsum(SingletSingletAggregation)) # https://stackoverflow.com/questions/33883758/python-sum-all-previous-values-in-array-at-each-index
plt.xlabel('Time(s)')
plt.ylabel('Event Count (Cumulative)')
plt.savefig(filename+'.singlet_collisions_cumulative.png') #print the plot

plt.figure()
plt.plot(TList[:-1],SingletDoubletAggregation)
plt.plot(TList[:-N],np.convolve(SingletDoubletAggregation, np.ones((N,))/N, mode='valid')) #https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
plt.xlabel('Time(s)')
plt.ylabel('Event Count')
plt.savefig(filename+'.doublet_singlet_collisions_average.png') #print the plot

plt.figure()
plt.plot(TList[:-1],np.cumsum(SingletDoubletAggregation)) # https://stackoverflow.com/questions/33883758/python-sum-all-previous-values-in-array-at-each-index
plt.xlabel('Time(s)')
plt.ylabel('Event Count (Cumulative)')
plt.savefig(filename+'.doublet_singlet_collisions_cumulative.png') #print the plot

plt.figure()
plt.plot(TList[:-N],(np.convolve(SingletSingletAggregation, (FrameSampleRate/(fps*tBrown))*np.ones((N,))/N, mode='valid'))/(FrameAreaRadii*Oned[:-N]*Oned[:-N])) #https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
plt.xlabel('Time(s)')
plt.ylabel('(Event Count/s)/Area*n1^2 ~ K11')
plt.savefig(filename+'.K11_collisions_average_normed.png') #print the plot

plt.figure()
plt.plot(TList[:-N],(np.convolve(SingletDoubletAggregation, (FrameSampleRate/(fps*tBrown))*np.ones((N,))/N, mode='valid'))/(FrameAreaRadii*Oned[:-N]*Twod[:-N])) #https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
plt.xlabel('Time(s)')
plt.ylabel('(Event Count/s)/Area*n1*n2 ~ K12')
plt.savefig(filename+'.K12_collisions_average_normed.png') #print the plot

plt.figure()
plt.plot(TList[:-N],(np.convolve(SingletTripletAggregation, (FrameSampleRate/(fps*tBrown))*np.ones((N,))/N, mode='valid'))/(FrameAreaRadii*Oned[:-N]*Threed[:-N])) #https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
plt.xlabel('Time(s)')
plt.ylabel('(Event Count/s)/Area*n1*n3 ~ K13')
plt.savefig(filename+'.K13_collisions_average_normed.png') #print the plot

#from joblib import Parallel, delayed
#import multiprocessing
#num_cores = multiprocessing.cpu_count()

#collect the pair distance information
for ttt in xrange(0,PeOneSamples+1): # loop through the times starting at the distance of Peclet One and ending at the collision
    Pairf.write("\n"+str(TList[ttt]))
    Pair11f.write("\n"+str(TList[ttt]))
    Pair12f.write("\n"+str(TList[ttt]))
    Pair13f.write("\n"+str(TList[ttt]))
    PairCount=0 # used to print to a specific location in PairData
    for kkk in xrange(0,len(KeyTList)): # loop through the groups in the key list
        #if(len(KeyTList[kkk])>2): Pairf.write(",") # only print groups that have more than one particle
        for pppB in xrange(1,len(KeyTList[kkk])-1): # loop through the particles that are partners in the group
            #print("ttt,kkk,pppB",ttt,kkk,pppB)
            timeTemp=ttt+KeyTList[kkk][pppB][0]-PeOneSamples
            Pairf.write(",")
            particleA=KeyTList[kkk][0][1]
            particleB=KeyTList[kkk][pppB][1]
            if(masses[particleA]+masses[particleB]==2):
                Pair11f.write(",")
            elif(masses[particleA]+masses[particleB]==3):
                Pair12f.write(",")
            elif((masses[particleA]==1 and masses[particleB]==3) or (masses[particleA]==3 and masses[particleB]==1)):
                Pair13f.write(",")
            if(len(Centroids)>timeTemp and timeTemp>0):
                #print("particleA",particleA,"Centroids[timeTemp][particleA]",Centroids[timeTemp][particleA])
                #print("particleB",particleB,"Centroids[timeTemp][particleB]",Centroids[timeTemp][particleB])
                xPairDisSq=(Centroids[timeTemp][particleA][0]-Centroids[timeTemp][particleB][0])**2 # the squared-x distance between one particle and its partner
                yPairDisSq=(Centroids[timeTemp][particleA][1]-Centroids[timeTemp][particleB][1])**2 # the squared-y distance between one particle and its partner
                disTemp=math.sqrt(xPairDisSq+yPairDisSq)*MicronPerPix
                Pairf.write(str(disTemp))
                if(masses[particleA]+masses[particleB]==2):
                    Pair11f.write(str(disTemp))
                    #print("particleA",particleA,"particleB",particleB,"timeTemp Pair data:",timeTemp)
                elif(masses[particleA]+masses[particleB]==3):
                    Pair12f.write(str(disTemp))
                elif((masses[particleA]==1 and masses[particleB]==3) or (masses[particleA]==3 and masses[particleB]==1)):
                    Pair13f.write(str(disTemp))
                if(yPairDis>0):
                    angleTemp=abs(round(math.atan(xPairDis/yPairDis)*180/math.pi))
                else:
                    angleTemp=0
                if(masses[particleA]+masses[particleB]==2):
                    PairData[PairCount].append(disTemp)
                    PairDataX[PairCount].append(math.sqrt(xPairDisSq))
                    PairDataY[PairCount].append(math.sqrt(yPairDisSq))
                elif(masses[particleA]+masses[particleB]==3):
                    PairData_12[PairCount].append(math.sqrt(xPairDisSq))
                elif((masses[particleA]==1 and masses[particleB]==3) or (masses[particleA]==3 and masses[particleB]==1)):
                    PairData_13[PairCount].append(math.sqrt(xPairDisSq))
            PairCount += 1
            #print("timeTemp",timeTemp,"ttt",ttt,"KeyTList[kkk][pppB][0]",KeyTList[kkk][pppB][0],"distance",math.sqrt(xPairDisSq+yPairDisSq),"particleA",particleA,"particleB",particleB)
#print(PairData)
#print("KeyTList",len(KeyTList),KeyTList)
for kkk in xrange(0,len(KeyTList)): # loop through the groups in the key list
    KeyTList[kkk].sort(reverse=True) # https://wiki.python.org/moin/HowTo/Sorting # sort from last time to first
    #print("kkk, KeyTList[kkk]", kkk,KeyTList[kkk])
    massPar=0 # number of particles in the aggregate
    Tm1=int(len(Centroids)-1) # the time step to compare to current time
    for pppB in xrange(0,len(KeyTList[kkk])-1): # loop through the particles that are partners in the group
        massPar+=masses[KeyTList[kkk][pppB][1]] # number of particles in the aggregate, will update as the numbers change while going backwards in the time count
    for pppB in xrange(1,len(KeyTList[kkk])-1): # loop through the significant times # moving average
        TimeTemp=TList[KeyTList[kkk][pppB-1][0]+1]-TList[KeyTList[kkk][pppB][0]-1]-PeOneSec # the total time segment in seconds, 2 frames are lost due to the collision even drastically shifting the center of mass/location of the chain
        TimeTemp=max(TimeTemp,PeOneSec)
        timeID=KeyTList[kkk][pppB-1][0]
        if(timeID-1-PeOneFrames>-1): # make sure there are at least 3 frames between collision times plus the amount of frames required for brownian/magnetic regime switch
            timeID=KeyTList[kkk][pppB-1][0]
            timeTot=TList[KeyTList[kkk][pppB][0]-PeOneSamples]-TList[KeyTList[kkk][pppB][0]] #time(s) covered over the collision
            parID=KeyTList[kkk][pppB-1][1]
            particleA=KeyTList[kkk][0][1] # index particle to compare to the others
            particleB=KeyTList[kkk][pppB][1] # particle to compare to the index
            MSDTemp=(Centroids[timeID-1-PeOneFrames][particleB][0]-Centroids[timeID+1][particleB][0])**2+(Centroids[timeID-1-PeOneFrames][particleB][1]-Centroids[timeID+1][particleB][1])**2 # the Mean Squared Displacement to be divided by 4t to get the diffusion coefficients
            DsTemp=MSDTemp*MicronPerPix**2/(4*TimeTemp) # the current diffusion coefficient
            #print("particleB",particleB,"TimeTemp",TimeTemp,"MicronPerPix**2/(4*TimeTemp)",MicronPerPix**2/(4*TimeTemp),"MSDTemp",MSDTemp,"DsTemp",DsTemp)
            DsList[massPar].append(DsTemp) # a collection of all the diffusion constants
            KeyTList[kkk][pppB].append([massPar,round(DsTemp*100000)/100000.0])
            #print("kkk",kkk,"pppB",pppB,"timeID",timeID,"DsTemp",DsTemp,"timeID-1-PeOneFrames",Centroids[timeID-1-PeOneFrames][particleB],"timeID+1",Centroids[timeID+1][particleB])
            if(pppB>1): # if this is not the final time, there was a collision which can be used to calculate the dipole strength
                #C5sTemp=((Centroids[timeID-1-PeOneFrames][particleA][0]-Centroids[timeID-1-PeOneFrames][particleB][0])**2+(Centroids[timeID-1-PeOneFrames][particleA][1]-Centroids[timeID-1-PeOneFrames][particleB][1])**2)**2.5/timeTot
                C5sTemp=MSDTemp**2.5/PeOneSec
                #print("particleA,particleB,C5sTemp",particleA,particleB,C5sTemp)
                #C5s[massPar][masses[KeyTList[kkk][pppB][1]]].append(C5sTemp)
                KeyTList[kkk][pppB].append([massPar,masses[KeyTList[kkk][pppB][1]],round(C5sTemp)])
        massPar-=masses[KeyTList[kkk][pppB][1]] # remove the mass of the particle being aggregated from the total mass
    # for the final key time, there is no aggregation, so there is no need for the complex separation of diffusion and dipole aggregation
    timeID_f=KeyTList[kkk][-2][0]
    timeID_i=KeyTList[kkk][-1][0]
    TimeTemp=TList[timeID_f]-TList[timeID_i]
    particleA=KeyTList[kkk][0][1] # index particle to compare to the others
    MSDTemp=(Centroids[timeID_f][particleA][0]-Centroids[timeID_i][particleA][0])**2+(Centroids[timeID_f][particleA][1]-Centroids[timeID_i][particleA][1])**2 # the Mean Squared Displacement, all particles are the same so we choose particle "0"
    if(TimeTemp>0):
        DsTemp=MSDTemp*MicronPerPix**2/(4*TimeTemp) # the current diffusion coefficient
    else:
        DsTemp=0
    DsList[massPar].append(DsTemp) # a collection of all the diffusion constants
    KeyTList[kkk][0].append([massPar,round(DsTemp*100000)/100000.0])
    #print("particleA",particleA,"TimeTemp",TimeTemp,"MicronPerPix**2/(4*TimeTemp)",MicronPerPix**2/(4*TimeTemp),"MSDTemp",MSDTemp,"DsTemp",DsTemp)
    #print("TimeTemp",TimeTemp,"timeID_i",timeID_i,"timeID_f",timeID_f,"DsTemp",DsTemp,"KeyTList",KeyTList[kkk])
Pairf.close()
Pair11f.close()
Pair12f.close()
Pair13f.close()

Difff = open(filename+'.Diffusion.csv','w') #open a text file to Diffusion constants
Difff.write("Number of Particles,Diffusion Constant,Samples,") # The average diffusion constant
for iii in xrange(1,100):
    Difff.write("C5(microns^5/s),dipolei*dipolej(A*micron^2),dipolei(A*micron^2) "+str(iii)+",Samples,")
Difff.write("\n")
for iii in xrange(1,len(DsList)):
    if(len(DsList[iii])>0):
        Difff.write(str(iii)+","+str(sum(DsList[iii])/len(DsList[iii]))+","+str(len(DsList[iii]))+",")
    else:
        Difff.write(str(iii)+",,,")
    mui=0 # dipole of the i particle
    for jjj in xrange(0,iii-1):
        Difff.write(",,,,")
    for jjj in xrange(iii,len(DsList)):
        muj="nan"
        if(jjj==iii):
            samples=len(C5s[iii][jjj])
            if(samples>0):
                SumSamples=sum(C5s[iii][jjj])
                MedianSamples=np.median(C5s[iii][jjj])
                mui_muj=abs(MedianSamples/Fivemu0__4pisq_mu_r) # product of the two dipoles
                mui=math.sqrt(mui_muj) # dipole of particle i
                muj=mui # dipole of particle j
        else:
            samples=len(C5s[iii][jjj])+len(C5s[jjj][iii]) # the number of samples
            if(samples>0):
                SumSamples=sum(C5s[iii][jjj])+sum(C5s[jjj][iii]) # the sum of the samples
                print("C5s[iii][jjj]+C5s[jjj][iii]",C5s[iii][jjj]+C5s[jjj][iii])
                MedianSamples=np.median(C5s[iii][jjj]+C5s[jjj][iii])
                mui_muj=abs(MedianSamples/Fivemu0__4pisq_mu_r) # product of the two dipoles
                if(mui>0):
                    muj=mui_muj/mui # dipole of particle j
        if(samples>0):
            Difff.write(str(MedianSamples)+","+str(mui_muj)+","+str(muj)+","+str(samples)+",")
        else: # if there are no samples of the C5 of this kind, put spaces
            Difff.write(",,")
    Difff.write("\n")
Difff.close()

calculationsf.write('Diffusion Coefficient from Stokes-Einstein equation in Micrometers^2 Per second'+','+str(DSE)+'\n') # http://en.wikipedia.org/wiki/Einstein_relation_(kinetic_theory)#Stokes-Einstein_equation # https://en.wikipedia.org/wiki/Viscosity#Water # (1.380*10^-23)*(J/K)*298*K/(6*pi*0.00089*Pa*s) in microns^3/s
calculationsf.write('The characteristic Brownian time in seconds'+','+str(tBrown)+'\n')
calculationsf.write('Average Squared Displacement in Micrometers^2 Per Second (calculated)'+','+str(SqrDisplacementMagTemp*MicronPerPix**2/TList[-1])+'\n')
calculationsf.write('Average Displacement in Micrometers Per Second (calculated)'+','+str(math.sqrt(SqrDisplacementMagTemp)*MicronPerPix/TList[-1])+'\n')
calculationsf.write('Average X Displacement in Micrometers Per Second'+','+str(DisplacementXTemp*MicronPerPix/TList[-1])+'\n')
calculationsf.write('Average Y Displacement in Micrometers Per Second'+','+str(DisplacementYTemp*MicronPerPix/TList[-1])+'\n')
if(len(DsList[1])>0):
    Deff=abs(sum(DsList[1]))/len(DsList[1]) # the effective diffusion of the singlets
    calculationsf.write('Diffusion Coefficient in Micrometers^2 Per second'+','+str(Deff)+'\n')
    Deff=(math.sqrt(Deff)-math.sqrt(SqrDisplacementMagTemp*MicronPerPix**2/(4*TList[-1])))**2 # subtract the displacement from the mean squared displacement
    calculationsf.write('Effective Diffusion Coefficient in Micrometers^2 Per second'+','+str(Deff)+'\n')
    calculationsf.write('Velocity (microns/s)'+','+str(math.sqrt(4*abs(Deff-DSE)/TauR))+'\n')
calculationsf.write('Rotational Diffusion Time (s)'+','+str(TauR)+'\n') # http://www.wolframalpha.com/input/?i=8*pi*%28mPa*s%29*%281*microns%29%5E3%2F%284.11*10%5E-21*J%29 # http://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.088304 #https://en.wikipedia.org/wiki/Rotational_diffusion#Basic_equations_of_rotational_diffusion
calculationsf.write('Number of particles: '+','+str(nParticles)+'\n')
#if(len(TList)>3 and TList[1]>0 and TList[-1]>0): calculationsf.write('Slope of Log-Log Length vs time: '+','+str(math.log(LengthList[-1]/LengthList[1])/math.log(TList[-1]/TList[1]))+'\n')
SurfaceConc=nParticles/(heightV*widthV*MicronPerPix**2)
calculationsf.write('Height(pix), Width(pix): '+','+str(heightV)+','+str(widthV)+'\n')
calculationsf.write('Height(radii), Width(radii): '+','+str(heightV*MicronPerPix/ParRMicrons)+','+str(widthV*MicronPerPix/ParRMicrons)+'\n')
calculationsf.write('Frame Area in Radii: '+','+str(FrameAreaRadii)+'\n')
calculationsf.write('Surface Concentration (particle/micron^2): '+','+str(SurfaceConc)+'\n')
calculationsf.write('Initial Fraction of Singlets: '+','+str(TZeroList[0]/nParticles)+'\n')
calculationsf.write('Surface Area Fraction: '+','+str(SurfaceAreaFraction)+'\n')
calculationsf.write('Volume Concentration (particle/micron^3): '+','+str(nParticles/(150*heightV*widthV*MicronPerPix**2))+'\n')
calculationsf.write('Time Elapsed(s): '+','+str(frameno*1.0/fps)+'\n')
calculationsf.write('Frames per Second: '+','+str(fps)+'\n')
calculationsf.write('Total Fames '+','+str(frameno)+'\n')
calculationsf.write('Time when all particles were singlets (seconds)'+','+str(tMinus)+'\n')
#calculationsf.write('Average Distance Between Nearest Neighbors (microns and frame 1)'+','+str(AverageDis*MicronPerPix)+'\n')
print("Area Pixels: ",FrameAreaPixels," , Area Microns: ",FrameAreaMicrons," , Area Radii: ",FrameAreaRadii)
calculationsf.write('Area Pixels'+','+str(FrameAreaPixels)+'\n')
calculationsf.write('Area Microns'+','+str(FrameAreaMicrons)+'\n')
calculationsf.write('Area Radii'+','+str(FrameAreaRadii)+'\n')
calculationsf.write('K11(r^2/par*t)'+','+str(newratesAbs[0])+'\n')
calculationsf.write('K12'+','+str(newratesAbs[1])+'\n')
calculationsf.write('K13'+','+str(newratesAbs[2])+'\n')
calculationsf.write('K22'+','+str(newratesAbs[3])+'\n')
calculationsf.write('K23'+','+str(newratesAbs[4])+'\n')
calculationsf.write('K33'+','+str(newratesAbs[5])+'\n')
calculationsf.close() # close the calculations file


#capC = cv2.VideoCapture(filename+".simple.avi")
NumberOfImages=0-1
for framenoK in xrange(0, FinalFrame, FrameSampleRate):
    NumberOfImages+=1
    #capC.set(CV_CAP_PROP_POS_FRAMES,int(framenoK))
    #ret, frame = capC.read()
    frame = cv2.imread(filename+'.Counting.'+(str(int(framenoK)).zfill(6))+'.png',0)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for ttt in xrange(1,len(Centroids)): # loop through the times
        for parA in xrange(0,len(Centroids[ttt])): # loop through the particles
            #cv2.putText(frame,str(parA), (Centroids[ttt][parA][0],Centroids[ttt][parA][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
            #cv2.putText(frame,str(parA), (Centroids[ttt-1][parA][0],Centroids[ttt-1][parA][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
            cv2.line(frame, (Centroids[ttt][parA][0],Centroids[ttt][parA][1]), (Centroids[ttt-1][parA][0],Centroids[ttt-1][parA][1]), Colors[parA],1)
    cv2.imwrite(filename+'.trajectory.'+(str(int(framenoK)).zfill(6))+'.png',frame)
os.system(ffmpegCommand+".trajectory.*.png' -y "+filename+".trajectory.gif")
# ffmpeg -framerate 1 -i Fe1-xO_batch020717_2.4_PS_5x_030717_128p1-clipped.avi.trajectory.%06d.png -y Fe1-xO_batch020717_2.4_PS_5x_030717_128p1-clipped.avi.trajectory.gif
os.system("cp "+filename+".trajectory.000000.png "+filename+".trajectory.png") # the first frame of the trajectory
FinalFrameZfill = (str(int(NumberOfImages*FrameSampleRate)).zfill(6))
#os.system("cp "+filename+".trajectory."+FinalFrameZfill+".png "+filename+".trajectory.png") # the final frame of the trajectory
os.system("rm "+filename+".trajectory.*.png") # make a pic of the first image

trajectoryf = open(filename+'.trajectory.html','w') #open a text file to write box locations
trajectoryf.write("<html><body>")
trajectoryf.write("<img id='myImage' src='"+str(filename)+".trajectory.gif' alt='Map'usemap='#planetmap'/> <map name='planetmap' id='map'>\n\n")
for group in xrange(0,len(KeyTList)): # loop through the groups
    Stitle=""
    for parA in xrange(len(KeyTList[group])-1,-1,-1): # loop through the particles in the group
        Stitle+=" &#xA;Particle ID: "+str(KeyTList[group][parA][1]) # the particle ID
        #print("parA",parA,"TList[KeyTList[group][parA]]",TList[KeyTList[group][parA][0]],"KeyTList[group]",KeyTList[group])
        if(parA==0):
            Stitle+=" &#xA;Time(s):"+str(round(TList[KeyTList[group][parA][0]],1))+",Loc(px):"+str(KeyTList[group][parA][2])+","+str(KeyTList[group][parA][3]) # print the time in seconds and the location of the first and last locations
        else:
            Stitle+=" &#xA;Time(s):"+str(round(TList[KeyTList[group][parA][0]],1))+",Loc(px):"+str(KeyTList[group][parA][2])
            if(len(KeyTList[group][parA])>3):
                #print("TList[KeyTList[group][parA]",TList[KeyTList[group][parA][0]],"KeyTList[group][parA]",KeyTList[group][parA])
                Stitle+=",D:"+str(KeyTList[group][parA][3])
                if(len(KeyTList[group][parA])>4):
                    Stitle+=",Constant:"+str(KeyTList[group][parA][4])
    trajectoryf.write("<area shape='circle' coords='"+str(KeyTList[group][-1][2][0])+","+str(KeyTList[group][-1][2][1])+",5' title='"+str(Stitle)+"' />\n")
trajectoryf.write("\n\n</map></body></html>")
trajectoryf.close() #http://www.w3schools.com/js/tryit.asp?filename=tryjs_imagemap http://stackoverflow.com/questions/10243606/imagemap-onmouseover-solution # next line: &#xA; or &#013;

#os.system("ffmpeg -framerate 1 -i 'concat:"+filename+".Counting."+FinalFrameZfill+".png|"+filename+".trajectory.png' -y "+filename+".FinalFrames.gif")



os.system("ffmpeg -framerate 1 -i 'concat:"+filename+".Counting."+FinalFrameZfill+".png|"+filename+".trajectory.png' -y "+filename+".FinalFrames.gif")


os.system("rm "+filename+".simple.*.png")
os.system("rm "+filename+".Edges.*.png")
os.system("rm "+filename+".Counting.*.png")
#os.system('tar -cf my_archive.tar $( find -maxdepth 1 -name *.csv" -or -name "*.png" )') # make an archive of small files to easily transport the data #https://stackoverflow.com/questions/18731603/how-to-tar-certain-file-types-in-all-subdirectories
#os.system("tar -jcvf archive"+filename+".tar.bz2 *.csv") # make an archive of small files to easily transport the data #https://stackoverflow.com/questions/18731603/how-to-tar-certain-file-types-in-all-subdirectories


for group in xrange(0,len(KeyTList)): # loop through the groups
    for parA in xrange(len(KeyTList[group])-1,-1,-1): # loop through the particles in the group
        if(len(KeyTList[group][parA])>4):
            #print(KeyTList[group][parA][3])
            #CollisionCount[KeyTList[group][parA][0]][KeyTList[group][parA][4][0]][KeyTList[group][parA][4][1]]+=1
            pass
            #print("KeyTList[group][parA][4][0]",KeyTList[group][parA][4][0],KeyTList[group][parA][4][1])
for iii in xrange(0,len(CollisionCount)):
    if(max(CollisionCount[iii][0])>0 or max(CollisionCount[iii][1])>0 or max(CollisionCount[iii][2])>0):
        pass
        #print(iii)
        #print(CollisionCount[iii][0])
        #print(CollisionCount[iii][1])
        #print(CollisionCount[iii][2])


C5_11=np.median(C5s[1][1])
if(str(C5_11)!="nan"):
    plt.figure()
    for xxx in xrange(0,len(PairData)):
        plt.plot(np.array(TList[0:len(PairData[xxx])])+PeOneSec-TList[len(PairData[xxx])-1],PairData[xxx])
    mu1_mu1=abs(C5_11/Fivemu0__4pisq_mu_r) # product of the two dipoles
    mu1=math.sqrt(mu1_mu1) # dipole of particle i
    plt.plot(np.array(TList[0:PeOneSamples+2])[::-1],(C5_11*np.array(TList[0:PeOneSamples+2]))**0.2, 'r--', linewidth=3, label='Fit: C5_11='+str(int(C5_11))+",m1="+str(round(mu1,4)))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (microns)')
    plt.legend(bbox_to_anchor=(0., 1.0, 1., .10),loc=3, borderaxespad=0., ncol=2)
    plt.savefig(filename+'.Pairs_Common_Collision_Pe_11.png') #print the plot of the particles all colliding at Pe time (from Pe=1 to collision)

    plt.figure()
    #f = open(filename+'.PairDataXY.csv','r') #open a text file to read
    #f.write("x-microns,y-microns,\n")
    for xxx in xrange(0,len(PairDataX)):
        #f.write(str(PairDataX[xxx])+","+str(PairDataY[xxx])+"\n")
        plt.plot(PairDataX[xxx],PairDataY[xxx],'r-o', linewidth=3)
        #f.close()
    #plt.plot(np.array(TList[0:PeOneSamples+2])[::-1],(C5_11*np.array(TList[0:PeOneSamples+2]))**0.2, 'r--', linewidth=3, label='Fit: C5_11='+str(int(C5_11))+",m1="+str(round(mu1,4)))
    plt.xlabel('Distance-X (microns)')
    plt.ylabel('Distance-Y (microns)')
    #plt.legend(bbox_to_anchor=(0., 1.0, 1., .10),loc=3, borderaxespad=0., ncol=2)
    plt.savefig(filename+'.Pairs_Common_Collision_Pe_XY.png') #print the plot of the particles all colliding at Pe time (from Pe=1 to collision)
else:
    print("No Singlet-Singlet Aggregation")

C5_12=np.median(C5s[1][2]+C5s[2][1])
if(str(C5_12)!="nan"):
    plt.figure()
    for xxx in xrange(0,len(PairData_12)):
        plt.plot(np.array(TList[0:len(PairData_12[xxx])])+PeOneSec-TList[len(PairData_12[xxx])-1],PairData_12[xxx])
    mu1_mu2=abs(C5_12/Fivemu0__4pisq_mu_r) # product of the two dipoles
    mu2=mu1_mu2/mu1 # dipole of particle i
    plt.plot(np.array(TList[0:PeOneSamples+2])[::-1],(C5_12*np.array(TList[0:PeOneSamples+2]))**0.2, 'r--', linewidth=3, label='Fit: C5_12='+str(int(C5_12))+",m1="+str(round(mu2,4)))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (microns)')
    plt.legend(bbox_to_anchor=(0., 1.0, 1., .10),loc=3, borderaxespad=0., ncol=2)
    plt.savefig(filename+'.Pairs_Common_Collision_Pe_12.png') #print the plot of the particles all colliding at Pe time (from Pe=1 to collision)
else:
    print("No Singlet-Doublet Aggregation")

C5_13=np.median(C5s[1][3]+C5s[3][1])
if(str(C5_13)!="nan"):
    plt.figure()
    for xxx in xrange(0,len(PairData_13)):
        plt.plot(np.array(TList[0:len(PairData_13[xxx])])+PeOneSec-TList[len(PairData_13[xxx])-1],PairData_13[xxx])
    mu1_mu3=abs(C5_13/Fivemu0__4pisq_mu_r) # product of the two dipoles
    mu3=mu1_mu3/mu1 # dipole of particle i
    plt.plot(np.array(TList[0:PeOneSamples+2])[::-1],(C5_13*np.array(TList[0:PeOneSamples+2]))**0.2, 'r--', linewidth=3, label='Fit: C5_13='+str(int(C5_13))+",m1="+str(round(mu3,4)))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (microns)')
    plt.legend(bbox_to_anchor=(0., 1.0, 1., .10),loc=3, borderaxespad=0., ncol=2)
    plt.savefig(filename+'.Pairs_Common_Collision_Pe_13.png') #print the plot of the particles all colliding at Pe time (from Pe=1 to collision)
else:
    print("No Singlet-Triplet Aggregation")


print("Process Completed Without Failure")




"""
from sys import platform
if platform == "linux" or platform == "linux2": # linux
    os.system("mpirun -np 2 lammps -in in.dipole_colloids_load_file")
elif platform == "darwin": # OS X
    os.system("mpirun -np 2 lammps -in in.dipole_colloids_load_file")
elif platform == "win32": # Windows... 
    os.system("lammps < in.dipole_colloids_load_file")
    pass
os.system("python Coordination_Number_Analysis.py "+str(filename))
"""

with open("time(s)_frames_framesPerSec.csv", "a") as myfile: # http://stackoverflow.com/questions/4706499/how-do-you-append-to-a-file-in-python
    myfile.write('\n'+str(time.time()-startTime)+','+str(len(Td))+','+str(len(Td)/(time.time()-startTime)))