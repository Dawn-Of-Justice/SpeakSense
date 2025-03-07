import os
import subprocess
from shutil import rmtree
import os
import sys
import time
import pickle
import glob
import cv2
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
import tqdm
import warnings
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

import sys, time, torch, glob, python_speech_features
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from model.faceDetector.s3fd import S3FD
from ASD import ASD
warnings.filterwarnings("ignore")
# C:\Users\Rohit Francis\Desktop\Codes\TESTINGS\Diarization Test\Light-ASD\model\faceDetector\s3fd\box_utils.py:104:
#     UserWarning: An output with one or more elements was resized since it had shape [54], which does not match the required output shape [33]. 
#     This behavior is deprecated, and in a future PyTorch release outputs will not be resized unless they have zero elements. You can explicitly reuse an out tensor t by resizing it,
#     inplace, to zero elements with t.resize_(0). (Triggered internally at
#                                                   C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\Resize.cpp:35.)

class SimpleConfig:
    def __init__(self):
        # Basic paths
        self.videoPath = None  # Will be set by user
        self.savePath = None   # Will be set by user
        
        # Derived paths (will be set automatically)
        self.pyaviPath = None
        self.pyframesPath = None
        self.videoFilePath = None
        self.audioFilePath = None
        self.pyworkPath = None
        self.pycropPath = None
        
        # Model related paths
        self.pretrainModel = "weight/pretrain_AVA_CVPR.model"  # Path to pretrained model
        
        # Processing parameters
        self.nDataLoaderThread = 10
        self.facedetScale = 0.25
        self.minTrack = 10
        self.numFailedDet = 10
        self.minFaceSize = 1
        self.cropScale = 0.40
        
        self.videoFolder = r"C:\Users\Rohit Francis\Desktop\Codes\TESTINGS\Diarization Test\Light-ASD\VideoFolder"
        
        # Video timing
        self.start = 0
        self.duration = 0  # 0 means process entire video

# The rest of your functions remain the same, just change 'args' to 'config'
def scene_detect(config):
    # Your existing scene_detect function, using config instead of args
    videoManager = VideoManager([config.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(config.pyworkPath, 'scene.pckl')
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
    with open(savePath, 'wb') as fil:
        pickle.dump(sceneList, fil)
    return sceneList

def inference_video(config):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cpu')
	flist = glob.glob(os.path.join(config.pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[config.facedetScale])
		dets.append([])
		for bbox in bboxes:
			dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		sys.stderr.write('%s-%05d; %d dets\r' % (config.videoFilePath, fidx, len(dets[-1])))
	savePath = os.path.join(config.pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


def track_shot(config, sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= config.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > config.minTrack:
			frameNum    = np.array([ f['frame'] for f in track ])
			bboxes      = np.array([np.array(f['bbox']) for f in track])
			frameI      = np.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = np.stack(bboxesI, axis=1)
			if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > config.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def crop_video(config, track, cropFile):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(config.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = config.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		image = cv2.imread(flist[frame])
		frame = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	# f'ffmpeg -y -i "{config.videoFilePath}" -ac 1 -ar 16000 "{config.audioFilePath}"'
	command = f'ffmpeg -y -i "{config.audioFilePath}" -ac 1 -acodec pcm_s16le -ar 16000 -ss {audioStart:.3f} -to {audioEnd:.3f} "{audioTmp}"'
	# command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
	# 	      (config.audioFilePath, config.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = f'ffmpeg -y -i "{cropFile}t.avi" -i "{audioTmp}" -c:v copy -c:a copy "{cropFile}.avi"'

	# command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
	# 		  (cropFile, audioTmp, config.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}

def extract_MFCC(file, outPath):
	# CPU: extract mfcc
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	np.save(featuresPath, mfcc)

def evaluate_network(files, config):
	# GPU: active speaker detection by pretrained model
	s = ASD()
	s.loadParameters(config.pretrainModel)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%config.pretrainModel)
	s.eval()
	allScores = []
	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	for file in tqdm.tqdm(files, total = len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
		_, audio = wavfile.read(os.path.join(config.pycropPath, fileName + '.wav'))
		audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
		video = cv2.VideoCapture(os.path.join(config.pycropPath, fileName + '.avi'))
		videoFeature = []
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break
		video.release()
		videoFeature = np.array(videoFeature)
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * 25)),:,:]
		allScore = [] # Evaluation use model
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
					inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
					embedA = s.model.forward_audio_frontend(inputA)
					embedV = s.model.forward_visual_frontend(inputV)	
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		allScore = np.round((np.mean(np.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)	
	return allScores

def visualization(tracks, scores, config):
	# CPU: visulize the result for video format
	flist = glob.glob(os.path.join(config.pyframesPath, '*.jpg'))
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
			s = np.mean(s)
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	firstImage = cv2.imread(flist[0])
	fw = firstImage.shape[1]
	fh = firstImage.shape[0]
	vOut = cv2.VideoWriter(os.path.join(config.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
	colorDict = {0: 0, 1: 255}
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		image = cv2.imread(fname)
		for face in faces[fidx]:
			clr = colorDict[int((face['score'] >= 0))]
			txt = round(face['score'], 1)
			cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
			cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
		vOut.write(image)
	vOut.release()
	# command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
	# 	(os.path.join(config.pyaviPath, 'video_only.avi'), os.path.join(config.pyaviPath, 'audio.wav'), \
	# 	config.nDataLoaderThread, os.path.join(config.pyaviPath,'video_out.avi')))
	command = f'ffmpeg -y -i "{os.path.join(config.pyaviPath, "video_only.avi")}" -i "{os.path.join(config.pyaviPath, "audio.wav")}" -c:v copy -c:a copy "{os.path.join(config.pyaviPath, "video_out.avi")}"'
	output = subprocess.call(command, shell=True, stdout=None)

def evaluate_col_ASD(tracks, scores, config):
    # config.videoFolder = config.colSavePath
	txtPath = config.videoFolder + '/col_labels/fusion/*.txt' # Load labels
	
	predictionSet = {}
	for name in {'long', 'bell', 'boll', 'lieb', 'sick', 'abbas'}:
		predictionSet[name] = [[],[]]
	dictGT = {}
	txtFiles = glob.glob("%s"%txtPath)
	for file in txtFiles:
		lines = open(file).read().splitlines()
		idName = file.split('/')[-1][:-4]
		for line in lines:
			data = line.split('\t')
			frame = int(int(data[0]) / 29.97 * 25)
			x1 = int(data[1])
			y1 = int(data[2])
			x2 = int(data[1]) + int(data[3])
			y2 = int(data[2]) + int(data[3])
			gt = int(data[4])
			if frame in dictGT:
				dictGT[frame].append([x1,y1,x2,y2,gt,idName])
			else:
				dictGT[frame] = [[x1,y1,x2,y2,gt,idName]]	
	flist = glob.glob(os.path.join(config.pyframesPath, '*.jpg')) # Load files
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]				
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = np.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]) # average smoothing
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		if fidx in dictGT: # This frame has label
			for gtThisFrame in dictGT[fidx]: # What this label is ?
				faceGT = gtThisFrame[0:4]
				labelGT = gtThisFrame[4]
				idGT = gtThisFrame[5]
				ious = []
				for face in faces[fidx]: # Find the right face in my result
					faceLocation = [int(face['x']-face['s']), int(face['y']-face['s']), int(face['x']+face['s']), int(face['y']+face['s'])]
					faceLocation_new = [int(face['x']-face['s']) // 2, int(face['y']-face['s']) // 2, int(face['x']+face['s']) // 2, int(face['y']+face['s']) // 2]
					iou = bb_intersection_over_union(faceLocation_new, faceGT, evalCol = True)
					if iou > 0.5:
						ious.append([iou, round(face['score'],2)])
				if len(ious) > 0: # Find my result
					ious.sort()
					labelPredict = ious[-1][1]
				else:					
					labelPredict = 0
				x1 = faceGT[0]
				y1 = faceGT[1]
				width = faceGT[2] - faceGT[0]
				predictionSet[idGT][0].append(labelPredict)
				predictionSet[idGT][1].append(labelGT)
	names = ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas'] # Evaluate
	names.sort()
	F1s = 0
	for i in names:
		scores = np.array(predictionSet[i][0])
		labels = np.array(predictionSet[i][1])
		scores = np.int64(scores > 0)
		F1 = f1_score(labels, scores)
		ACC = accuracy_score(labels, scores)
		if i != 'abbas':
			F1s += F1
			print("%s, ACC:%.2f, F1:%.2f"%(i, 100 * ACC, 100 * F1))
	print("Average F1:%.2f"%(100 * (F1s / 5)))	  




def preprocess_video_file(video_file, output_root):
    """
    Simplified video preprocessing that:
    1. Converts video to standard AVI format
    2. Extracts audio as WAV
    3. Extracts frames as JPG images
    
    Parameters:
        video_file (str): Path to input video file
        output_root (str): Root directory for outputs
    """
    # Initialize configuration
    config = SimpleConfig()
    config.videoPath = video_file
    config.savePath = os.path.join(output_root, 'results')
    
    # Set up directory structure
    config.pyaviPath = os.path.join(config.savePath, 'pyavi')
    config.pyframesPath = os.path.join(config.savePath, 'pyframes')
    config.videoFilePath = os.path.join(config.pyaviPath, 'video.avi')
    config.audioFilePath = os.path.join(config.pyaviPath, 'audio.wav')
    config.pyworkPath = os.path.join(config.savePath, 'pywork')
    config.pycropPath = os.path.join(config.savePath, 'pycrop')
    
    # Create fresh directories
    if os.path.exists(config.savePath):
        rmtree(config.savePath)
    os.makedirs(config.pyaviPath)
    os.makedirs(config.pyframesPath)
    os.makedirs(config.pyworkPath)
    os.makedirs(config.pycropPath)
    
    print("Processing video:", config.videoPath)
    
    # 1. Convert video to standard AVI format (25 fps)
    print("Converting video to AVI format...")
    command = f'ffmpeg -y -i "{config.videoPath}" -qscale:v 2 -r 25 "{config.videoFilePath}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in video conversion:", result.stderr)
        return None
    print("Video converted successfully")
    
    # 2. Extract audio as WAV (16kHz, mono)
    print("Extracting audio...")
    command = f'ffmpeg -y -i "{config.videoFilePath}" -ac 1 -ar 16000 "{config.audioFilePath}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in audio extraction:", result.stderr)
        return None
    print("Audio extracted successfully")
    
    # 3. Extract frames as JPG images
    print("Extracting frames...")
    command = f'ffmpeg -y -i "{config.videoFilePath}" -qscale:v 2 -f image2 "{os.path.join(config.pyframesPath, "%06d.jpg")}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in frame extraction:", result.stderr)
        return None
    print("Frames extracted successfully")
    
    return {
        'video': config.videoFilePath,
        'audio': config.audioFilePath,
        'frames': config.pyframesPath
    }, config

if __name__ == "__main__":
    # Example usage
    # video_file = r"path/to/your/video.mp4"  # Replace with your video path
    # output_dir = r"path/to/output"          # Replace with your output path
    
    video_file = r"C:\Users\Rohit Francis\Desktop\Codes\TESTINGS\Diarization Test\Light-ASD\ASD_SAMPLE - Made with Clipchamp.mp4"
    output_dir = r"C:\Users\Rohit Francis\Desktop\Codes\TESTINGS\Diarization Test\Light-ASD\PreprocessOutput"
    
    result, config = preprocess_video_file(video_file, output_dir)
    
    if result:
        print("\nProcessing completed successfully!")
        print(f"Processed video: {result['video']}")
        print(f"Extracted audio: {result['audio']}")
        print(f"Extracted frames: {result['frames']}")
    else:
        print("Processing failed!")
        
    scene = scene_detect(config)
    print(scene)
    
    faces = inference_video(config)
    # print(faces)
    
    allTracks = []
    for shot in scene:
        # Check if the shot length is sufficient for tracking
        if shot[1].frame_num - shot[0].frame_num >= config.minTrack:
            # faces[shot[0].frame_num:shot[1].frame_num] corresponds to the frames in this scene
            tracks_in_shot = track_shot(config, faces[shot[0].frame_num:shot[1].frame_num])
            allTracks.extend(tracks_in_shot)

    # print(allTracks[0])
    
    vidTracks = []
    files = os.listdir(r"C:\Users\Rohit Francis\Desktop\Codes\TESTINGS\Diarization Test\Light-ASD\PreprocessOutput\results\pycrop")
    for ii, track in enumerate(tqdm.tqdm(allTracks, desc="Cropping face clips")):
        # print("inside loop:" ,ii, track)
        crop_file_base = os.path.join(config.pycropPath, f'{ii:05d}')
        cropped_result = crop_video(config, track, crop_file_base)
        vidTracks.append(cropped_result)
  
    # for ii, track in enumerate(tqdm.tqdm(allTracks, desc="Cropping face clips")):
    #     try:
    #         print("inside loop:" ,ii, files[ii])
    #         # crop_file_base = os.path.join(config.pycropPath, f'{ii:05d}')

    #         cropped_result = crop_video(config, track, files[ii])
    #         vidTracks.append(cropped_result)

    #     except IndexError as e:
    #         print(vidTracks)

    # print(vidTracks)

    # Save the tracks (optional, for later inspection)
    tracks_save_path = os.path.join(config.pyworkPath, 'tracks.pckl')
    with open(tracks_save_path, 'wb') as fil:
        pickle.dump(vidTracks, fil)
        
    fil = open(tracks_save_path, 'rb')
    vidTracks = pickle.load(fil)
    
    # print("Vidtracks: ")
    # print(vidTracks)
    
    files = glob.glob("%s/*.avi"%config.pycropPath)
    files.sort()
    scores = evaluate_network(files, config)
    print("scores: ", scores)
    
    savePath = os.path.join(config.pyworkPath, 'scores.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(scores, fil)
    print(scores[0].shape)    
    evaluate_col_ASD(vidTracks, scores, config)
    visualization(vidTracks, scores, config)