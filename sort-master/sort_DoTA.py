"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from sort import *
from PIL import Image, ImageDraw, ImageFont
import pickle

def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print("Error: Creating directory." + directory)

def xy_limit(x1,y1):
    if x1 < 0:
        x1 = 0
    if x1 > 1280:
        x1 = 1280
    if y1 < 0 :
        y1 = 0
    if y1 > 720:
        y1 = 720
    return x1,y1

def xy2cxcy(x1,y1,x2,y2):
    w= x2 - x1
    h = y2 -y1
    cx = x1 + w/2
    cy = y1 + h/2
    return cx,cy,w,h


def vis_sort_result(image_path, save_path, frame, trackers):
    image= Image.open(image_path)
    draw = ImageDraw.Draw(image)
    fnt =  ImageFont.truetype('C:/Windows/Fonts/Arial.ttf' ,30)

    for d in trackers:
        [x1, y1, x2, y2, id] = d
        x1,y1 = xy_limit(x1,y1)
        x2, y2 = xy_limit(x2,y2)
        draw.rectangle(((x1,y1),(x2,y2)), outline=(255,0,0), width=2)   #visualization position, size, color
        draw.text((x1,y1), str(int(id)), font=fnt, fill=(0,255,0))
    image.save(save_path)


for f_num in range(5):
    f_n = f_num + 1
    # detection_result_folder = "E:/DoTA/dataset/ego_involved/0_test/yolov3_result/"
    # frame_folder = "E:/DoTA/dataset/ego_involved/0_test/frames/"
    # save_path = "E:/DoTA/dataset/ego_involved/0_test/sort_result_yolov3/"
    # detection_result_folder = "G:/Intern/mask_rcnn_result_kor/%d/"%f_n
    # frame_folder = "G:/Intern/frames/%d/"%f_n
    # save_path = "G:/Intern/sort_result/%d/"%f_n
    # save_vis = "G:/Intern/frame_id/%d/"%f_n

    detection_result_folder = "D:/DAD/mask_rcnn_result/train/positive/"
    frame_folder = "E:/DAD/frames/train/positive/"
    save_path = "E:/DAD/sort_result/train/positive/"

    createFolder(save_path)
    # createFolder(save_vis)

    dataset_list = os.listdir(detection_result_folder)
    #print(dataset_list)
    progress = 1
    total = len(dataset_list)
    for dataset in dataset_list:
        foldername = dataset.split(".")[0]
        print(">> Foldername:", foldername, "(%d /%d)"%(progress,total))
        progress += 1

        # img_list = os.listdir(frame_folder + foldername +"/images/")
        img_list = os.listdir(frame_folder + foldername)


        createFolder(save_path + foldername)
        createFolder(save_path + foldername + "/visualization")
        # createFolder(save_vis + foldername)

        #create instance of SORT
        mot_tracker = Sort()

        total_frames = 0
        object_id_list =[]
        listall =[]
        frameid_list = []

        #get detections information
        seq_dets = np.loadtxt(detection_result_folder + dataset, delimiter=",")
        if seq_dets.shape ==(10,):
            seq_dets=seq_dets.reshape((1,10))

        for frame in range(len(img_list)):
            frame +=1 # start frame id is 1.

            # image_path = frame_folder +foldername+ "/images/"+ "{:06}".format(frame) +".jpg"
            image_path = frame_folder +foldername+ "/"+ "{:06}".format(frame) +".jpg"
            save_img_path = save_path + foldername +"/visualization/" + "{:06}".format(frame) +".jpg"
            # save_img_path = save_vis + foldername + "/" + "{:06}".format(frame) + ".jpg"

            #print(" frame : ", frame)
            if seq_dets[seq_dets[:,0]==frame] == []:
                #Nothing Detection
                vis_original_img(image_path, save_img_path)

            else:
                dets = seq_dets[seq_dets[:,0]==frame, 2:7]
                dets[:,2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]

                trackers = mot_tracker.update(dets)

                for d in trackers:
                    [x1, y1, x2, y2, id] = d
                    id = int(id)
                    x1,y1 = xy_limit(x1,y1)
                    x2,y2 = xy_limit(x2,y2)
                    cx,cy,w,h = xy2cxcy(x1,y1,x2,y2)

                    if not id in object_id_list :
                        #this tracking id is first time.
                        object_id_list.append(id)
                        frameid_list.append([frame])
                        listall.append([[cx,cy,w,h]])

                    else:
                        index = object_id_list.index(id)
                        temp = listall.pop(index)
                        temp.insert(len(temp),[cx,cy,w,h])
                        listall.insert(index, temp)

                        temp2 = frameid_list.pop(object_id_list.index(id))
                        temp2.append(frame)
                        frameid_list.insert(object_id_list.index(id), temp2)

                vis_sort_result(image_path, save_img_path, frame, trackers)

        mot_tracker.reset_count()


        result_path = save_path + foldername
        for k in range(len(object_id_list)):
            #result_file = result_path + "/test_"+str(nth_dataset)+"_obj_"+str(object_id_list[k])+".pkl"
            result_file = result_path + "/" + str(foldername) + "_"+str(object_id_list[k])+".pkl"
            result = {'bbox' : np.asarray(listall[k]),'frame_id' : np.asarray(frameid_list[k])}
            with open(result_file, 'wb') as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)