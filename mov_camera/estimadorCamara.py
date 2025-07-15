import cv2
import pickle
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance

class EstimadorMovimientoCam():
    def __init__(self,frame):
        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS or cv2.TERM_CRITERIA_COUNT, 10, 0.05)
        )
        
        self.minDistance = 5
        
        firstFrameGrayScale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        maskFeatures = np.zeros_like(firstFrameGrayScale)
        maskFeatures[:,0:20] = 1
        maskFeatures[:,900:1050] = 1
        
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = maskFeatures
        )
    
    def add_adjust_position_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frameNum, track in enumerate(object_tracks):
                for trackId, trackInfo in track.items():
                    position = trackInfo['position']
                    cameraMovement = camera_movement_per_frame[frameNum]
                    positionAdjusted = (position[0]-cameraMovement[0],position[1]-cameraMovement[1])
                    tracks[object][frameNum][trackId]['position_adjusted'] = positionAdjusted
        
    def get_camera_movement(self,frames,readFromStub=False, stubPath=None):
        
        if readFromStub and stubPath is not None and os.path.exists(stubPath):
            with open(stubPath,'rb') as f:
                return pickle.load(f)
        
        cameraMovement = [[0,0]]*len(frames)
        
        oldGray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        oldFeatures = cv2.goodFeaturesToTrack(oldGray,**self.features)
        
        for frameNum in range(1,len(frames)):
            frameGray = cv2.cvtColor(frames[frameNum],cv2.COLOR_BGR2GRAY)
            newFeatures, _, _ = cv2.calcOpticalFlowPyrLK(oldGray, frameGray, oldFeatures, None, **self.lk_params)
            
            maxDistance = 0
            cameraMov_x ,cameraMov_y = 0,0
            
            for i, (new,old) in enumerate(zip(newFeatures,oldFeatures)):
                newFeatures_point = new.ravel()
                oldFeatures_point = old.ravel()
                
                distance = measure_distance(newFeatures_point,oldFeatures_point)
                
                if distance > maxDistance:
                    maxDistance = distance
                    cameraMov_x = oldFeatures_point[0] - newFeatures_point[0]
                    cameraMov_y = oldFeatures_point[1] - newFeatures_point[1]
                    
            if maxDistance > self.minDistance:
                cameraMovement[frameNum] = [cameraMov_x,cameraMov_y]
                oldFeatures = cv2.goodFeaturesToTrack(frameGray,**self.features)
                
            oldGray = frameGray.copy()
            
        if stubPath is not None:
            with open(stubPath,'wb') as f:
                pickle.dump(cameraMovement,f)
            
        return cameraMovement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames=[]
        
        for frameNum, frame in enumerate(frames):
            frame = frame.copy()
            
            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,80), (255,255,255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
            
            xMov, yMov = camera_movement_per_frame[frameNum]
            frame = cv2.putText(frame,f"Movimiento de camara X: {xMov: 2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,0),2)
            frame = cv2.putText(frame,f"Movimiento de camara Y: {yMov: 2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,0),2)
            
            output_frames.append(frame)
            
        return output_frames