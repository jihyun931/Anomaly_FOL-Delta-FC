# Anomaly_FOL-Delta-FC
 
1)
Mask RCNN
C:\Users\user\Desktop\Anomaly_Detection\IROS2019\mask_rcnn\maskrcnn-benchmark
실행파일 excute_mask_rcnn.py
수정 사항
line 28 : dataset_path
line 72 : save_path
line 34 : 읽기 path의 경우 이미지가 저장된 폴더 주소까지 넣어줘야함
data/DoTA/frames/images/이미지.jpg 의 경우 "/images" 까지 주소로 넣음

mask rcnn 결과 형식
[frame id, -1, x, y, w, h, score, -1,-1,-1]
frame id의 경우 한 프레임에 여러 오브젝트가 탐지될 경우 중복될 수 있음

python excute_mask_rcnn.py


2)
Sort (Deep sort)
C:\Users\user\Desktop\Anomaly_Detection\IROS2019\deep_sort\sort-master
실행파일 sort_DoTA.py
수정사항
line64~66
detection_result_folder
frame_folder
save_path
line83,106 위와 마찬가지로 "/images/" 까지 주소 필요 (/images/ 와 /images 구분 주의)

python sort_DoTA.py


########### optical flow 미사용 시 불필요 ##################################
3)
flownet2
C:\Users\user\Desktop\Anomaly_Detection\IROS2019\flownet2-pytorch
실행파일 run_flownet_DoTA.py
수정사항
line 46,47
folder_path
flow_folder_path
line 55,65,66
마찬가지로 이미지 경로 (/images/)

4)
roi pooling
C:\Users\user\Desktop\Anomaly_Detection\IROS2019\flownet2-pytorch
실행파일 roi_pooling_DoTA.py
수정사항
line 71~74
main_folder
flow_folder_path
sort_folder_path
pooling_path
line 90 마찬가지로 /images/ 경로

####################################################################


############# train part (test 시 불필요) ############################################
5)
train dataset 생성
C:\Users\user\Desktop\Anomaly_Detection\IROS2019\FOL_custom\ap_train_dataset\DoTA_based\ap_train_dataset
실행파일 make_ap_train_dataset.py (this.this. 써진거)
수정사항
line 115~120
positive_dataset_folder
negative_dataset_folder
save_folder
p_data_num
n_data_num
tv_ratio
line 72 : 필요에 따라 anotation 파일 변경
line 107 : 필요에 따라 xlsx 파일 변경
anotation 파일 :
anomaly_start/end,frame_id, image_path, accident_id, accident_name, objects 정보


6)
get_fol_hidden_state_concat.py
C:\Users\user\Desktop\Anomaly_Detection\IROS2019\FOL_custom
실행파일 get_fol_hidden_state_concat.py
수정사항
line 108 : save_path (train, val 수정해서 따로 뽑음)
config --load_config config/get_fol_hidden_state_concat.yaml
data_root

7)
train_AP_with_AdaLEA_concat.py
C:\Users\user\Desktop\Anomaly_Detection\IROS2019\FOL_custom
실행파일 train_AP_with_AdaLEA_concat.py
수정사항
line 70 : writer -> summary 저장 경로
config --load_config config/train_ap_AdaLEA.yaml
data_root
checkpoint_dir

python train_AP_with_AdaLEA_concat.py--load_config config/train_ap_AdaLEA.yaml


####################################################################


########### test part  ###################################

8)
eval_obj_AP_on_DoTA.py
C:\Users\user\Desktop\Anomaly_Detection\IROS2019\FOL_custom
실행파일 eval_obj_AP_on_DoTA.py
수정사항
line177,178
test_dataset_path
save_path
line 18, 164
import FolRnned_wo_ego

config --load_config config/eval_obj_AP_concat.yaml
best_fol_model
best_ap_model

python eval_object_AP_on_DoTA.py --load_config config/eval_obj_AP_concat.yaml



9)
map_attc_each_obj_DoTA.py (each는 obj별 map_attc_DoTA는 영상 자체)
C:\Users\user\Desktop\Anomaly_Detection\IROS2019\FOL_custom
실행파일 map_attc_each_obj_DoTA.py
수정사항
line 114 : main_path

python map_attc_each_obj_DoTA.py


####################################################################

======== FOL part train ===================

1)
C:\Users\Yoon\Desktop\work\LAB\Anomaly\model\Anticipate_Traffic_Accident-master
실행파일 train_fol_absolute_vanilla.py
수정사항
line 76,77
summary 경로

config --load_config config/train_fol_absolute_vanilla.yaml
data_root
checkpoint_dir
line 30
pred_dim 4 <-> 6



2)
C:\Users\Yoon\Desktop\work\LAB\Anomaly\model\Anticipate_Traffic_Accident-master
실행파일 test_fol_absolute_vanilla.py
수정사항
config --load_config config/test_fol_absolute_vanilla.yaml
data_root
best_fol_model

line 30
pred_dim 4 <-> 6
