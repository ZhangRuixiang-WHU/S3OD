
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially6.sh semi 5 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 5 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially4.sh semi 1 1 1

# for FOLD in 1 2 3 4 5;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi ${FOLD} 1 1
# done

# for FOLD in 1 2 3 4 5;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi ${FOLD} 5 1
# done

# for FOLD in 1 2 3 4 5;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh semi ${FOLD} 1 1
# done


# for FOLD in 5;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially6.sh semi ${FOLD} 10 1
# done


# for FOLD in 2 3 4 5;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi ${FOLD} 1 1
# done
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh semi 2 10 1
# for FOLD in 1 2 3 4 5;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially4.sh semi ${FOLD} 10 1
# done

# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh baseline 100 100 1

# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh semi 100 100 1

# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh baseline 100 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 100 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh semi 100 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh semi 5 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi 1 1 1


# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh baseline 100 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh baseline 100 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh baseline 100 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh baseline 100 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh baseline 100 100 1
# for FOLD in 5 4;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh semi ${FOLD} 10 1
# done

# for FOLD in 1 2 5;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially4.sh semi ${FOLD} 10 1
# done


# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh semi 1 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially6.sh semi 1 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh semi 1 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 1 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh semi 1 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially6.sh semi 1 100 1 
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 1 100 1 
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi 1 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi 1 100 1 
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh semi 1 100 1 
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi 1 100 1 

# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh baseline 100 100 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh baseline 100 100 1

# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially4.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially4.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh semi 1 1 1


# cp -r /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-1@1/labels/* /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-1@1/pseudo_labels_STAC/
# cp -r /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-2@1/labels/* /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-2@1/pseudo_labels_STAC/
# cp -r /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-3@1/labels/* /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-3@1/pseudo_labels_STAC/
# cp -r /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-4@1/labels/* /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-4@1/pseudo_labels_STAC/
# cp -r /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-5@1/labels/* /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-5@1/pseudo_labels_STAC/
# cp -r /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-1@5/labels/* /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-1@5/pseudo_labels_STAC/
# cp -r /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-2@5/labels/* /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-2@5/pseudo_labels_STAC/
# cp -r /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-3@5/labels/* /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-3@5/pseudo_labels_STAC/
# cp -r /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-4@5/labels/* /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-4@5/pseudo_labels_STAC/
# cp -r /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-5@5/labels/* /data/zrx/Rotated/DOTA-v2/coco/semi_train/semi-5@5/pseudo_labels_STAC/


# for FOLD in 1 2 3 4 5;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh baseline ${FOLD} 1 1
# done

# for FOLD in 1 2 3 4;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh baseline ${FOLD} 5 1
# done

# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partiallyb.sh semi 2 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh semi 1 1 1

CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh baseline 1 1 1
CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh baseline 1 1 1
# for FOLD in 3 ;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi ${FOLD} 10 1
# done

# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh baseline 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi 1 1 1
# for FOLD in 4 5;
# do
#   CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi ${FOLD} 5 1
# done
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh baseline 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially6.sh baseline 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi 1 1 1

# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh baseline 1 1 1
# CUDA_VISIBLE_DEVICES=1 bash tools/dist_train_partially3.sh baseline 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially4.sh semi 100 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially4.sh baseline 100 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially6.sh semi 100 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially2.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially4.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially5.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially6.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially.sh semi 1 1 1
# CUDA_VISIBLE_DEVICES=0 bash tools/dist_train_partially3.sh semi 1 1 1

# python tools/test.py work_dirs/S3OD_100precent_240k/1/S3OD_100precent_240k.py work_dirs/S3OD_100precent_240k/1/iter_320000.pth --eval mAP
# python tools/test.py work_dirs/S3OD_100precent_240k/1/S3OD_100precent_240k_1024.py work_dirs/S3OD_100precent_240k/1/iter_240000.pth --eval mAP
# python tools/test.py work_dirs/as_s3od_35_directly/1/1/as_s3od_35_directly_1024.py work_dirs/as_s3od_35_directly/1/1/iter_96000.pth --eval mAP