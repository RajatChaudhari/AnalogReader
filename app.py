#!flask/bin/python
from flask import Flask, request, jsonify
import json
import os
from flask_cors import CORS
import numpy as np
import cv2
import datetime
import time
import logging
from logging.handlers import RotatingFileHandler
from threading import Thread

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

cors = CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "*"}})

"""
    Date: 2-June-2019
    Build:0.1
    Developer: rajat.chaudhari@accenture.com
    Functionality: reads analog gauge images and gives current reading
    Reviewer:
"""

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread. Used for live feed
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    def start(self):    
        Thread(target=self.get, args=()).start()
        #self.get()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
        self.stream.release()
        cv2.destroyAllWindows()



def dist_2_pts(x1, y1, x2, y2):
    #print np.sqrt((x2-x1)^2+(y2-y1)^2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def crop_cricle(image):
    """
    This is used to get the region of interest in cases of heavily tilted images, obsolete as it takes very long
    """
    """(h, w) = image.shape[:2]
    if (w>h) : 
        #M = cv2.getRotationMatrix2D((h/2,w/2), 270, 1.0)
        #image = cv2.warpAffine(image, M, (w,w))
        image = imutils.rotate_bound(image, 90)"""

    
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #convert to gray
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 80, 50, int(height*0.20), int(height*0.45))
    a, b, c = circles.shape
    x,y,r = avg_circles(circles, b)
    #draw center and circle
    #cv2.circle(image, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    #cv2.circle(image, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle
    icircle=circles[0,:][0]
    cropSize = (r*2, r*2)
    nx=x-int(icircle[0]-cropSize[1]//2)
    ny=y-int(icircle[1]-cropSize[0]//2)

    cropCoords = (max(0, icircle[1]-cropSize[0]//2),min(image.shape[0], icircle[1]+cropSize[0]//2),
                      max(0, icircle[0]-cropSize[1]//2),min(image.shape[1], icircle[0]+cropSize[1]//2)) 
    crop_cimg = image[int(cropCoords[0]):int(cropCoords[1]),int(cropCoords[2]):int(cropCoords[3])] 

    return crop_cimg,nx,ny,r

def alignImages(app,img1, img2,gauge_number, file_type,log_dir,debug):
    
    MAX_FEATURES = 1000#500 #
    GOOD_MATCH_PERCENT = 0.10#0.15 #
    #print("ALIGNING")
    grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    impts1, d1 = orb.detectAndCompute(grey1, None)
    impts2, d2 = orb.detectAndCompute(grey2, None)

    # Match features.
    featurematcheObj = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matchedfeatures = featurematcheObj.match(d1, d2, None)

    # Sort matches by score
    matchedfeatures.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    GoodMatches = int(len(matchedfeatures) * GOOD_MATCH_PERCENT)
    matchedfeatures = matchedfeatures[:GoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matchedfeatures), 2), dtype=np.float32)
    points2 = np.zeros((len(matchedfeatures), 2), dtype=np.float32)

    for i, feature in enumerate(matchedfeatures):
      points1[i, :] = impts1[feature.queryIdx].pt
      points2[i, :] = impts2[feature.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = img2.shape
    transformed = cv2.warpPerspective(img1, h, (width, height))
    if debug=="True":
        imgMatches = cv2.drawMatches(img1, impts1, img2, impts2, matchedfeatures, None)
        cv2.imwrite('%s/frame%s-align.%s' % (log_dir,gauge_number, file_type), transformed)
        cv2.imwrite('%s/frame%s-matches.%s' % (log_dir,gauge_number, file_type), imgMatches)
    return transformed, h

def calibrate_gauge(app,img,gauge_number, file_type,log_dir,debug):
    '''
        This function calibrates the range available to the dial as well as the
        units.  It works by first finding the center point and radius of the gauge.  Then it draws lines at intervals
        (separation) in degrees from values defined a config file. This assumes that the gauge is linear (as most probably are).
    '''
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray
    #detect circles
    #restricting the search from 35-48% of the possible radii gives fairly good results across different samples.  Remember that
    #these are pixel values which correspond to the possible radii search range.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 200, 80, int(height*0.35), int(height*0.48))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    a, b, c = circles.shape
    x,y,r = avg_circles(circles, b)
    #draw center and circle
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)
    if debug =="True":    
        #for calibration, plot lines from center going out at every 10 degrees and add marker
        #for i from 0 to 36 (every 10 deg)
        '''
        goes through the motion of a circle and sets x and y values based on the set separation spacing.  Also adds text to each
        line.  These lines and text labels serve as the reference point for the user to enter
        NOTE: by default this approach sets 0/360 to be the +x axis (if the image has a cartesian grid in the middle), the addition
        (i+9) in the text offset rotates the labels by 90 degrees so 0/360 is at the bottom (-y in cartesian).  So this assumes the
        gauge is aligned in the image, but it can be adjusted by changing the value of 9 to something else.
        '''
        
        separation = 10.0 #in degrees
        interval = int(360 / separation)
        p1 = np.zeros((interval,2))  #set empty arrays
        p2 = np.zeros((interval,2))
        p_text = np.zeros((interval,2))
        for i in range(0,interval):
            for j in range(0,2):
                if (j%2==0):
                    p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
                else:
                    p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
        text_offset_x = 10
        text_offset_y = 5
        for i in range(0, interval):
            for j in range(0, 2):
                if (j % 2 == 0):
                    p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                    p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+27) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
                else:
                    p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                    p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+27) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees
                    
        #add the lines and labels to the image
        for i in range(0,interval):
            cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 3)
            cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),2,cv2.LINE_AA)
        #saving calibrated frames for logs to validate, debug and troubleshoot
        cv2.imwrite('%s/funcframe%s-calibration.%s' % (log_dir,gauge_number, file_type), img)    
    return img, x, y, r

def get_current_value(app,img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_number, file_type,log_dir,outputFolder,debug):
    new_value=0
    # preprocessing the image for finding lines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.bitwise_not(gray)
    binaryImg = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 15, -2)
    
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN, kernel)
    medianBlur = cv2.medianBlur(binaryImg, 5)
    guassianBlur = cv2.GaussianBlur(medianBlur, (5, 5), 0)
    
    if debug=="True":
        #saving processed frames for logs to validate, debug and troubleshoot
        cv2.imwrite('%s/funcframe%s-tempdst1.%s' % (log_dir,gauge_number, file_type), guassianBlur)
    # find lines
    minLineLength = 200 # this is critical, can be taken from config file if AI people are handling this
    maxLineGap = 0
    
    lines = cv2.HoughLinesP(image=guassianBlur, rho=1, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=0)  # rho is set to 3 to detect more lines, easier to get more then filter them out later
    
    # define a range for lines
    final_line_list = []
    line_length=[]
    
    if lines is not None:
        for i in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
                diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
                #set diff1 to be the smaller (closest to the center) of the two), makes the math easier
                if (diff1 > diff2):
                    temp = diff1
                    diff1 = diff2
                    diff2 = temp                
                line_length.append(dist_2_pts(x1, y1, x2, y2))
                    # add to final list
                final_line_list.append([x1, y1, x2, y2])
                #cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        longest=line_length.index(max(line_length))    #get the longest line    
    # assuming the longest line is the best one
    if(len(final_line_list)>0):
        
        x1 = final_line_list[longest][0]
        y1 = final_line_list[longest][1]
        x2 = final_line_list[longest][2]
        y2 = final_line_list[longest][3]

        #find the farthest point from the center to be what is used to determine the angle
        dist_pt_0 = dist_2_pts(x, y, x1, y1)
        dist_pt_1 = dist_2_pts(x, y, x2, y2)
        if (dist_pt_0 > dist_pt_1):
            x_angle = x1 - x
            y_angle = y - y1
            cv2.line(img, (x, y), (x1, y1), (0, 255, 0), 2)
        else:
            x_angle = x2 - x
            y_angle = y - y2
            cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 2)
            
        # take the arc tan of y/x to find the radian angle
        res = np.arctan(np.divide(float(y_angle), float(x_angle)))
        res = np.rad2deg(abs(res))
        
        if x_angle > 0 and y_angle > 0:  #in quadrant I
            final_angle = 90-res
        if x_angle > 0 and y_angle < 0:  #in quadrant II
            final_angle = 90+res
        if x_angle < 0 and y_angle < 0:  #in quadrant III
            final_angle = 270-res
        if x_angle < 0 and y_angle > 0:  #in quadrant IV
            final_angle = 270+res
        # calculates the reading based on the gauge range and calibrated values
        old_min = float(min_angle)
        old_max = float(max_angle)

        new_min = float(min_value)
        new_max = float(max_value)

        old_value = final_angle

        old_range = (old_max - old_min)
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min
        
        cv2.putText(img, str(round(new_value,2)), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)
        if debug =="True":
            cv2.imwrite('%s/frame%s-lines.%s' % (log_dir,gauge_number, 'jpg'), img) #get path from config
        
        #cv2.imwrite('%s/Output_%s.%s' % (outputFolder,gauge_number, 'jpg'), img) #prints all frames for video
        imgName=('Output_%s.%s' % (gauge_number, 'jpg'))
        return img,round(new_value,2),imgName #change response image between frame and transformed image here
    else:
        app.logger.info("No line was detected on the gauge")
        return 0,int(-1),0
        

def GetReadings(app,file,path,frames_to_process,refimg,min_angle,max_angle,min_value,max_value,units,typeoffile,logFolder,outputFolder,currTime,debug,camera):
    GaugeReadings=[]
    frameNumber=0
    #to make sure only 1st image and reading is written to output
    found=0
    iName="empty"
    
    try:
        #classifying input as looping pattern changes depending on type of input file
        if (file.split('.')[-1] in ['jpg','jpeg','png']):
            try:
                frame=cv2.imread(file)
                frameNumber+=1
                currTime = datetime.datetime.now().strftime("%d.%m.%y-%H.%M.%S")
                image_Id=str(frameNumber)+'_'+currTime
                image,h=alignImages(app,frame,refimg,frameNumber, typeoffile,logFolder,debug)
                fimg, xf, yf, rf = calibrate_gauge(app,image,frameNumber, typeoffile,logFolder,debug)
                resulting_frame,currentValue,imgName = get_current_value(app,fimg, min_angle, max_angle, min_value, max_value, xf, yf, rf, image_Id, typeoffile,logFolder,outputFolder,debug)
                if imgName !=str(0) and found==0:
                    cv2.imwrite('%s/Output_%s.%s' % (outputFolder,image_Id, 'jpg'), resulting_frame)
                    iName=imgName
                    found=1                
                if currentValue > -1:
                        GaugeReadings.append(currentValue)
                        app.logger.info("Latest frame gauge reading : "+ str(currentValue)+" "+units)
            except Exception as e :
                app.logger.error("Failure in reading gauge value")
                app.logger.error(str(e))
        elif (file.split('.')[-1] in ['avi','wmv','mp4']):
            cap=cv2.VideoCapture(file)
            if cap.isOpened():
                ret, frame = cap.read()
            else:
                ret=False
            #if user gives frames to process more than the video length take minimum of the two
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while ret==True and frameNumber<int(min(int(frames_to_process),length)):
                try:
                    currTime = datetime.datetime.now().strftime("%d.%m.%y-%H.%M.%S")
                    frameNumber+=1
                    image_Id=str(frameNumber)+'_'+currTime
                    ret, frame = cap.read()
                    
                    image,h=alignImages(app,frame,refimg,frameNumber, typeoffile,logFolder,debug)
                    fimg, xf, yf, rf = calibrate_gauge(app,image,frameNumber, typeoffile,logFolder,debug)
                    resulting_frame,currentValue,imgName = get_current_value(app,fimg, min_angle, max_angle, min_value, max_value, xf, yf, rf, image_Id, typeoffile,logFolder,outputFolder,debug)
                    #write only 1 reading to output folder
                    if imgName !=str(0) and found==0:
                        iName=imgName
                        cv2.imwrite('%s/Output_%s.%s' % (outputFolder,image_Id, 'jpg'), resulting_frame)
                        found=1
                    
                    #dont consider bad frames as it interferes with average and min value calculations
                    if currentValue > -1:
                        GaugeReadings.append(currentValue)
                        app.logger.info("Latest frame gauge reading : "+ str(currentValue)+" "+units)
                except Exception as e :
                    app.logger.error("Failure in reading gauge value")
                    app.logger.error(str(e))
                    pass
            cap.release()
        elif file.split('\\')[-1]== "Live_Feed":
            record=0
            video_getter = VideoGet(camera).start()
            while (True) and frameNumber<int(frames_to_process):
                try:
                    frame = video_getter.frame
                    #special functionality to show input video feed to help user to adjust the position of gauge
                    #needs extra while loop to enable recording frames on press of 's' key
                    if debug == "True":
                        cv2.namedWindow("Input Frame")
                        cv2.imshow("Input Frame",frame)
                        if (cv2.waitKey(1) == ord("s")):
                            record=1
                        while (record==1) and frameNumber<int(frames_to_process):
                            frame = video_getter.frame
                            currTime = datetime.datetime.now().strftime("%d.%m.%y-%H.%M.%S")
                            frameNumber+=1
                            image_Id=str(frameNumber)+'_'+currTime
                            if debug == "True":
                                cv2.imwrite('%s/Input%s.%s' % (outputFolder,image_Id, typeoffile), frame)
                            image,h=alignImages(app,frame,refimg,frameNumber, typeoffile,logFolder,debug)
                            fimg, xf, yf, rf = calibrate_gauge(app,image,frameNumber, typeoffile,logFolder,debug)
                            resulting_frame,currentValue,imgName = get_current_value(app,fimg, min_angle, max_angle, min_value, max_value, xf, yf, rf, image_Id, typeoffile,logFolder,outputFolder,debug)
                            if imgName !=str(0) and found==0:
                                iName=imgName
                                found=1
                            
                            if currentValue > -1:
                                GaugeReadings.append(currentValue)
                                app.logger.info("Latest frame gauge reading : "+ str(currentValue)+" "+units)
                        record=0
                    else:
                        currTime = datetime.datetime.now().strftime("%d.%m.%y-%H.%M.%S")
                        frameNumber+=1
                        image_Id=str(frameNumber)+'_'+currTime
                        if debug == "True":
                            cv2.imwrite('%s/Input%s.%s' % (outputFolder,image_Id, typeoffile), frame)
                        image,h=alignImages(app,frame,refimg,frameNumber, typeoffile,logFolder,debug)
                        fimg, xf, yf, rf = calibrate_gauge(app,image,frameNumber, typeoffile,logFolder,debug)
                        resulting_frame,currentValue,imgName = get_current_value(app,fimg, min_angle, max_angle, min_value, max_value, xf, yf, rf, image_Id, typeoffile,logFolder,outputFolder,debug)
                        if imgName !=str(0) and found==0:
                            iName=imgName
                            cv2.imwrite('%s/Output_%s.%s' % (outputFolder,image_Id, 'jpg'), resulting_frame)
                            found=1
                        
                        if currentValue > -1:
                            GaugeReadings.append(currentValue)
                            app.logger.info("Latest frame gauge reading : "+ str(currentValue)+" "+units)
                except Exception as e :
                    app.logger.info("ERROR:Failure in reading gauge value")
                    app.logger.info(str(e))
                    pass
            video_getter.stop()
            del video_getter
    except Exception as e :
        app.logger.info("ERROR:Failure in reading gauge value")
        app.logger.info(str(e))
    
    return GaugeReadings,frameNumber,iName


def RemoveOutliers(readingList):
    GR=sorted(readingList)
    q1, q3= np.percentile(GR,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr) 
    upper_bound = q3 +(1.5 * iqr)
    
    GaugeReadings=[item for item in readingList if item > lower_bound and item <upper_bound]
    return GaugeReadings

@app.route('/', methods=['GET','POST'])
def readDial():
    App_start_time = time.time()
    try:
        with open(os.getcwd()+'\\gaugeconfig.json','r') as json_config_file:
            gaugeconfig = json.load(json_config_file)
        with open(os.getcwd()+'\\code_view_mapping.json','r') as json_data_file:
            dataconfig = json.load(json_data_file)
    except Exception as e:
        print("failed to load config file, error is - %s" % e)
    try:
        if not (os.path.exists(os.getcwd()+gaugeconfig['log_dir'])):
            os.mkdir(os.getcwd()+gaugeconfig['log_dir'])
    except OSError as e:  
        print ("Creation of the directory %s failed" % e)

    currTime = datetime.datetime.now().strftime("%d.%m.%y-%H.%M.%S")
    if not (os.path.exists(os.getcwd()+gaugeconfig['log_dir']+currTime)):
        os.mkdir(os.getcwd()+gaugeconfig['log_dir']+currTime)
    loc_code = request.args.get("functional_loc_code")
    path = request.args.get("path")
    frames_to_process = request.args.get("frames_to_process")
    if frames_to_process is None:
        frames_to_process=gaugeconfig['frames_to_process']
        
    GaugeReadings=[]
    
    min_angle = gaugeconfig['calibration']['min_angle'] 
    max_angle = gaugeconfig['calibration']['max_angle'] 
    min_value = gaugeconfig['calibration']['min_value']
    max_value = gaugeconfig['calibration']['max_value']
    units = gaugeconfig['calibration']['units']
    typeoffile=gaugeconfig["type_of_image"]
    camera=gaugeconfig["camera"]
    debug=gaugeconfig['debug']
    outliers=gaugeconfig['remove_outliers']    
    if (loc_code and path and frames_to_process and loc_code in dataconfig):
        
        try:
            if not (os.path.exists(path)):
                os.mkdir(path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)
       
        logFolder=os.getcwd()+gaugeconfig['log_dir']+currTime
        outputFolder=path        
        refimg=cv2.imread(os.getcwd()+gaugeconfig['reference_image'])
        file=os.getcwd()+dataconfig[loc_code]
        
        log_file= logFolder+'\\GR_'+currTime+'.log'
        open(log_file, 'w').close()
        handler = RotatingFileHandler(log_file, maxBytes=1024*1024*100, backupCount=1)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        app.logger.setLevel(logging.INFO)
        app.logger.addHandler(handler)
        
        GaugeReadings,framesProcessed,imgName = GetReadings(app,file,path,frames_to_process,refimg,min_angle,max_angle,min_value,max_value,units,typeoffile,logFolder,outputFolder,currTime,debug,camera)
        now = datetime.datetime.now()
        OverallTime=time.time()-App_start_time
        if len(GaugeReadings)>0:
            
            if outliers=="True":
                GR=RemoveOutliers(GaugeReadings)
                Average=round(sum(GR)/len(GR),2)
            else:
                Average=round(sum(GaugeReadings)/len(GaugeReadings),2)
            
            minimum = min(GaugeReadings)
            maximum = max(GaugeReadings)
            reading=GaugeReadings[0]
            readingStatus=1
        else:
            Average=0
            minimum=0
            maximum=0
            reading="No Readings Captured"
            readingStatus=0
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        response = {
            "Date" : date_time,
            "Reading_Status" : readingStatus,
            "Measure" : {
                "First Frame Reading":reading,
                "Average" : round(Average,2),
                "Min" : round(minimum,2),
                "Max" :round(maximum,2),
            },
            "Frame_Details":
            {
            "Frames_Processed" : framesProcessed,
            "Frames_Success" : len(GaugeReadings),
            "Total Time":OverallTime
            },
            "Analysis_image_path" : str(os.getcwd())+"\\"+path+"\\"+imgName,
            }
    else:
        response = {"ERROR":"Inavlid Payload or invalid parameter values. Please pass the query parameters functional_loc_code, path and frames_to_process and their expected values."}
        app.logger.info("ERROR:Invalid Payload or invalid parameter values")
    return jsonify(response)          
            
if __name__ == '__main__':
    app.run()