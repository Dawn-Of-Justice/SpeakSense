Ground truth information for the Columbia dataset used in the paper
"Cross-modal Supervision for Learning Active Speaker Detection in Video"
by
Punarjay Chakravarty and Tinne Tuytelaars.

The video can be downloaded from https://youtu.be/6GzxbrO0DHM.

Ground truth is organized by folders named according to frame numbers.
Eg:
tracks_25000_35000 contains tracks for frames 25000 to 35000.
Tracks are stored as txt files: 0.txt, 1.txt, 2.txt, for the 3 speakers in frames 
25000 to 35000.

Each track's txt file contains bounding box and speak/non-speak info as follows:

0.txt in folder tracks_25000_35000 ...

framenum  TLx  TLy bounding-box-width=height  speak=1/non-speak=0
25472	0	48	143	0	
25473	0	48	143	0	
25474	0	48	144	0	
25475	0	48	145	0	
25476	0	48	146	0	

Track for this speaker has square bounding box with TL x,y coordinates 0,48 and width of the bounding box = 143 and is not speaking in frame 25472.

