print("Hi")
import sys, time, os, tqdm, torch, glob, subprocess, warnings, cv2, pickle, numpy, python_speech_features
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from ASD import ASD

warnings.filterwarnings("ignore")
print("hi")
# Configuration class instead of command line arguments
class Config:
    def __init__(self):
        # Basic paths
        self.videoName = "sample_video.mp4"  # Name of your input video file
        self.videoFolder = "demo"  # Folder containing your input video
        self.pretrainModel = "weight/pretrain_AVA_CVPR.model"  # Path to pretrained model
        
        # Processing parameters
        self.nDataLoaderThread = 10
        self.facedetScale = 0.25
        self.minTrack = 10
        self.numFailedDet = 10
        self.minFaceSize = 1
        self.cropScale = 0.40
        
        # Video timing
        self.start = 0
        self.duration = 0  # 0 means process entire video
        
        # Derived paths
        # self.savePath = os.path.join(self.videoFolder, self.videoName)
        # self.videoPath = os.path.join(self.videoFolder, self.videoName + '.mp4')  # Assuming MP4 format
        # self.pyaviPath = os.path.join(self.savePath, 'pyavi')
        # self.pyframesPath = os.path.join(self.savePath, 'pyframes')
        # self.pyworkPath = os.path.join(self.savePath, 'pywork')
        # self.pycropPath = os.path.join(self.savePath, 'pycrop')
        
        self.savePath = r"C:\Users\Rohit Francis\Desktop\Codes\TESTINGS\Diarization Test\Light-ASD\results" 
        self.videoPath = r"C:\Users\Rohit Francis\Desktop\Codes\TESTINGS\Diarization Test\Light-ASD\demo\sample_video.mp4"
        self.pyaviPath = os.path.join(self.savePath, 'pyavi')
        self.pyframesPath = os.path.join(self.savePath, 'pyframes')
        self.pyworkPath = os.path.join(self.savePath, 'pywork')
        self.pycropPath = os.path.join(self.savePath, 'pycrop')
        
        
        # Additional paths set during processing
        self.videoFilePath = None
        self.audioFilePath = None

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

# Note: The other functions remain the same, just change 'args' to 'config'
# ... (keeping all other functions as they are)

def inference_video(config):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cuda')
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
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > config.minFaceSize:
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
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (config.audioFilePath, config.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, config.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}

def extract_MFCC(file, outPath):
	# CPU: extract mfcc
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	numpy.save(featuresPath, mfcc)

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
		videoFeature = numpy.array(videoFeature)
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
		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
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
			s = numpy.mean(s)
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
	command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
		(os.path.join(config.pyaviPath, 'video_only.avi'), os.path.join(config.pyaviPath, 'audio.wav'), \
		config.nDataLoaderThread, os.path.join(config.pyaviPath,'video_out.avi'))) 
	output = subprocess.call(command, shell=True, stdout=None)

def evaluate_col_ASD(tracks, scores, config):
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
			s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]) # average smoothing
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
		scores = numpy.array(predictionSet[i][0])
		labels = numpy.array(predictionSet[i][1])
		scores = numpy.int64(scores > 0)
		F1 = f1_score(labels, scores)
		ACC = accuracy_score(labels, scores)
		if i != 'abbas':
			F1s += F1
			print("%s, ACC:%.2f, F1:%.2f"%(i, 100 * ACC, 100 * F1))
	print("Average F1:%.2f"%(100 * (F1s / 5)))	  


def main():
    # Create and configure settings
    config = Config()
    
    # Set your paths here
    # config.videoName = "sample_video.mp4"  # Change this to your video name without extension
    # config.videoFolder = ""  # Change this to your video folder path
    # config.videoPath = "video.mp4"  # Full path to your video file
    # config.pretrainModel = "weight/pretrain_AVA_CVPR.model"  # Path to your pretrained model
    
    # Create necessary directories
    if os.path.exists(config.savePath):
        rmtree(config.savePath)
    os.makedirs(config.pyaviPath, exist_ok=True)
    os.makedirs(config.pyframesPath, exist_ok=True)
    os.makedirs(config.pyworkPath, exist_ok=True)
    os.makedirs(config.pycropPath, exist_ok=True)

    print("Here")

    # Set derived paths
    config.videoFilePath = os.path.join(config.pyaviPath, 'video.avi')
    config.audioFilePath = os.path.join(config.pyaviPath, 'audio.wav')

    # Extract video
    if config.duration == 0:
        command = f"ffmpeg -y -i {config.videoPath} -qscale:v 2 -threads {config.nDataLoaderThread} -async 1 -r 25 {config.videoFilePath} -loglevel panic"
    else:
        command = f"ffmpeg -y -i {config.videoPath} -qscale:v 2 -threads {config.nDataLoaderThread} -ss {config.start} -to {config.start + config.duration} -async 1 -r 25 {config.videoFilePath} -loglevel panic"
    subprocess.call(command, shell=True, stdout=None)
    
    # Continue with the rest of your processing pipeline...
    # (The rest of the main function remains the same, just change 'args' to 'config')
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(config.videoFilePath))
	
	# Extract audio
    config.audioFilePath = os.path.join(config.pyaviPath, 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % (config.videoFilePath, config.nDataLoaderThread, config.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(config.audioFilePath))

	# Extract the video frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
		(config.videoFilePath, config.nDataLoaderThread, os.path.join(config.pyframesPath, '%06d.jpg'))) 
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(config.pyframesPath))

	# Scene detection for the video frames
    scene = scene_detect(config)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(config.pyworkPath))	

	# Face detection for the video frames
    faces = inference_video(config)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(config.pyworkPath))

	# Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= config.minTrack: # Discard the shot frames less than minTrack frames
            allTracks.extend(track_shot(config, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

	# Face clips cropping
    for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
        vidTracks.append(crop_video(config, track, os.path.join(config.pycropPath, '%05d'%ii)))
    savePath = os.path.join(config.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %config.pycropPath)
    fil = open(savePath, 'rb')
    vidTracks = pickle.load(fil)

    # Active Speaker Detection
    files = glob.glob("%s/*.avi"%config.pycropPath)
    files.sort()
    scores = evaluate_network(files, config)
    savePath = os.path.join(config.pyworkPath, 'scores.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(scores, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %config.pyworkPath)

    if config.evalCol == True:
        evaluate_col_ASD(vidTracks, scores, config) # The columnbia video is too big for visualization. You can still add the `visualization` funcition here if you want
        quit()
    else:
        # Visualization, save the result as the new video	
        visualization(vidTracks, scores, config)	


# (Assume that all necessary imports and the functions below are already defined:
#  - scene_detect
#  - inference_video
#  - bb_intersection_over_union
#  - track_shot
#  - crop_video
#  - [other functions if needed])
# Also assume that S3FD, ASD, and any other models are imported properly.

def preprocess_video_file(video_file, output_root):
    """
    Preprocess a video file into cropped face clips for active speaker detection.
    
    This function:
      1. Sets up a configuration.
      2. Extracts and converts the video to a standard AVI format.
      3. Extracts audio and frames.
      4. Runs scene detection.
      5. Runs face detection and tracking.
      6. Crops the face clips with synchronized audio.
    
    Parameters:
      video_file (str): Full path to the input video file.
      output_root (str): Root directory where all intermediate and result folders will be created.
    
    Returns:
      List[str]: Sorted list of file paths (AVI files) in the cropped folder (ready for evaluate_network).
    """
    # Create a new configuration and update paths
    config = Config()
    
    # Update video path to the provided file
    config.videoPath = video_file

    # Set output paths based on the given output_root
    config.savePath = os.path.join(output_root, 'results')
    config.pyaviPath = os.path.join(config.savePath, 'pyavi')
    config.pyframesPath = os.path.join(config.savePath, 'pyframes')
    config.pyworkPath = os.path.join(config.savePath, 'pywork')
    config.pycropPath = os.path.join(config.savePath, 'pycrop')
    
    # (Optional) If you have additional parameters (like evalCol), you can set them here.
    config.evalCol = False

    # Remove any existing output (be careful – this deletes the folder!)
    if os.path.exists(config.savePath):
        rmtree(config.savePath)
    os.makedirs(config.pyaviPath, exist_ok=True)
    os.makedirs(config.pyframesPath, exist_ok=True)
    os.makedirs(config.pyworkPath, exist_ok=True)
    os.makedirs(config.pycropPath, exist_ok=True)

    # Derived paths for video and audio extraction
    config.videoFilePath = os.path.join(config.pyaviPath, 'video.avi')
    config.audioFilePath = os.path.join(config.pyaviPath, 'audio.wav')

    # 1. Convert the input video to a standardized AVI format (25 fps)
    if config.duration == 0:
        command = f"ffmpeg -y -i {config.videoPath} -qscale:v 2 -threads {config.nDataLoaderThread} -async 1 -r 25 {config.videoFilePath} -loglevel panic"
    else:
        command = f"ffmpeg -y -i {config.videoPath} -qscale:v 2 -threads {config.nDataLoaderThread} -ss {config.start} -to {config.start + config.duration} -async 1 -r 25 {config.videoFilePath} -loglevel panic"
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Video extracted to {config.videoFilePath}\n")
    
    # 2. Extract audio from the standardized video
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" %
               (config.videoFilePath, config.nDataLoaderThread, config.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Audio extracted to {config.audioFilePath}\n")

    # 3. Extract frames from the video
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" %
               (config.videoFilePath, config.nDataLoaderThread, os.path.join(config.pyframesPath, '%06d.jpg')))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Frames extracted to {config.pyframesPath}\n")

    # 4. Run scene detection on the extracted frames
    scene = scene_detect(config)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Scene detection completed.\n")

    # 5. Run face detection on the frames using S3FD
    faces = inference_video(config)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Face detection completed.\n")

    # 6. Face tracking: group face detections into continuous tracks.
    allTracks = []
    for shot in scene:
        # Check if the shot length is sufficient for tracking
        if shot[1].frame_num - shot[0].frame_num >= config.minTrack:
            # faces[shot[0].frame_num:shot[1].frame_num] corresponds to the frames in this scene
            tracks_in_shot = track_shot(config, faces[shot[0].frame_num:shot[1].frame_num])
            allTracks.extend(tracks_in_shot)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" {len(allTracks)} face tracks detected.\n")

    # 7. Crop face clips from the video based on the face tracks.
    vidTracks = []
    for ii, track in enumerate(tqdm.tqdm(allTracks, desc="Cropping face clips")):
        crop_file_base = os.path.join(config.pycropPath, f'{ii:05d}')
        cropped_result = crop_video(config, track, crop_file_base)
        vidTracks.append(cropped_result)

    # Save the tracks (optional, for later inspection)
    tracks_save_path = os.path.join(config.pyworkPath, 'tracks.pckl')
    with open(tracks_save_path, 'wb') as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Face clips cropped and saved in {config.pycropPath}\n")

    # 8. Return the list of cropped AVI file paths (which will be used by evaluate_network)
    cropped_files = sorted(glob.glob(os.path.join(config.pycropPath, '*.avi')))
    return cropped_files

# Example usage:
if __name__ == '__main__':
    # Replace these paths with your actual video file and desired output directory.
    video_file_path = r"C:\Users\Rohit Francis\Desktop\Codes\TESTINGS\Diarization Test\Light-ASD\ASD_SAMPLE - Made with Clipchamp.mp4"
    output_directory = r"C:\Users\Rohit Francis\Desktop\Codes\TESTINGS\Diarization Test\Light-ASD\PreprocessOutput"
    
    # Preprocess the video file
    cropped_clip_files = preprocess_video_file(video_file_path, output_directory)
    
    # Print the list of preprocessed (cropped) files.
    print("Preprocessed video clips:")
    for clip in cropped_clip_files:
        print(clip)
    
    # Now you can pass 'cropped_clip_files' to evaluate_network:
    # scores = evaluate_network(cropped_clip_files, config) 
    # (Make sure that the same configuration or any necessary parameters are available.)



# if __name__ == '__main__':
#     main()
#     evaluate_network()

