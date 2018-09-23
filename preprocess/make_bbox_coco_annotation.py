import json
import pandas as pd
import numpy as np
import copy 
import cv2

def get_bounding_box(img_path,train_data):
    bounding_box_label = []
    csv_head = ['image_id','image_category','xmin','ymin','xmax','ymax']
    for labels in train_data:
        img_name,img_label = labels[0],labels[1]
        img = cv2.imread(img_path+img_name)
        img_shape = img.shape
        x_axis = []
        y_axis = []
        for label_axis in labels[2:]:
            label_axis = label_axis.split('_')
            label_axis = map(float, label_axis)
            if label_axis[0] > 0 and label_axis[1] > 0:
                x_axis.append(label_axis[0])
                y_axis.append(label_axis[1])
        xmin,xmax,ymin,ymax = min(x_axis),max(x_axis),min(y_axis),max(y_axis)
        xmin,xmax,ymin,ymax = max(xmin-5,1),min(xmax+5,img_shape[1]),max(ymin-5,1),min(ymax+5,img_shape[0]) 
        bounding_box_label.append([img_name,img_label,xmin,ymin,xmax,ymax])
    df = pd.DataFrame(bounding_box_label , columns = csv_head)
    df.to_csv('val_bounding_box.csv')
    
def make_bbox_augmentation(prefix, category, img_path, bbox_path):
    """
    make bbox csv file for augmentation dataset using original bbox csv file
    Test code:
     img_path = '/home/ygj/Software/Detectron/lib/datasets/data/FashionAI/train/Images/blouse_color_aug'
     bbox_path = '/home/ygj/EclipseWorkspace/FashionAI_keypoint/bounding_box.csv'
     make_bbox_augmentation('randomColor', 'blouse', img_path, bbox_path)
"""
    origin_bboxes = pd.read_csv(bbox_path).values[:7158]
    aug_bboxes = []
    csv_head = ['image_id','image_category','xmin','ymin','xmax','ymax']
    assert os.path.isdir(img_path), 'invalid image file path'
    for im_name in os.listdir(img_path):
        origin_im_name = 'Images/'+category+'/'+im_name[len(prefix)+1:]
        aug_bbox = origin_bboxes[origin_bboxes[:,1] == origin_im_name]
        aug_bbox[0,1] = 'Images/'+img_path.split('/')[-1]+'/'+im_name
        aug_bboxes.append(list(aug_bbox[0,1:]))
    df = pd.DataFrame(aug_bboxes, columns=csv_head)
    df.to_csv('blouse_aug_bbox.csv')
    print 'finished!'
    
def make_coco_json(img_path, bbox_path, category):
    """Transform FashionAI bbox.csv to COCO json format"""
    bbox_data = pd.read_csv(bbox_path).values # the first column is id
    img_labels = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
    label_indexes = []
    for name in img_labels:
        label_indexes.append(np.where(bbox_data==name)[0])
    category_id = img_labels.index(category)
    category_id_train = int(len(label_indexes[category_id])*0.8)
    
    # make training set dict
    coco_dict = {}
    coco_dict['type'] = 'instances'
    coco_dict['images'] = []
    coco_dict['categories'] = []
    coco_dict['annotations'] = []
    # make category sub-dataset json file
    coco_dict['categories'].append({'supercategory':'none', 'id':1, 'name': category})
    for label_index in label_indexes[category_id][:category_id_train]:
        img_name = bbox_data[label_index][1]
        img = cv2.imread(img_path+img_name)
        height, width = img.shape[0], img.shape[1] # row, column
        image_dict = {'id': label_index, 'file_name':img_name,\
                      'width':width, 'height': height}
        coco_dict['images'].append(image_dict)
        xmin, ymin, xmax, ymax = map(int, list(bbox_data[label_index][-4:]))
        polygons = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
        area = (xmax-xmin)*(ymax-ymin)
        # coco 'bbox' format is [x, y, w, h] not [xmin, ymin, xmax, ymax]
        ann_dict = {'segmentation':[polygons], 'area':area, 'iscrowd':0, 'ignore':0,\
                    'image_id':label_index, 'bbox':[xmin, ymin, xmax-xmin, ymax-ymin],\
                    'category_id':1, 'id':label_index} 
        coco_dict['annotations'].append(ann_dict)
    coco_dict_train = coco_dict
    
    # make validation set dict
    coco_dict = {}
    coco_dict['type'] = 'instances'
    coco_dict['images'] = []
    coco_dict['categories'] = []
    coco_dict['annotations'] = []
    # make category sub-dataset json file
    coco_dict['categories'].append({'supercategory':'none', 'id':1, 'name': category})
    for label_index in label_indexes[category_id][category_id_train:]:
        img_name = bbox_data[label_index][1]
        img = cv2.imread(img_path+img_name)
        height, width = img.shape[0], img.shape[1] # row, column
        image_dict = {'id': label_index, 'file_name':img_name,\
                      'width':width, 'height': height}
        coco_dict['images'].append(image_dict)
        xmin, ymin, xmax, ymax = map(int, list(bbox_data[label_index][-4:]))
        polygons = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
        area = (xmax-xmin)*(ymax-ymin)
        ann_dict = {'segmentation':[polygons], 'area':area, 'iscrowd':0, 'ignore':0,\
                    'image_id':label_index, 'bbox':[xmin, ymin, xmax-xmin, ymax-ymin],\
                    'category_id':1, 'id':label_index}
        coco_dict['annotations'].append(ann_dict)
    coco_dict_val = coco_dict
    
    return coco_dict_train, coco_dict_val

def make_warm_up_json(img_path, bbox_path, category):
    """make validation set with warm_up data"""
    bbox_data = pd.read_csv(bbox_path).values # the first column is id
    img_labels = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
    label_indexes = []
    for name in img_labels:
        label_indexes.append(np.where(bbox_data==name)[0])
    category_id = img_labels.index(category)
    # make validation set dict
    coco_dict = {}
    coco_dict['type'] = 'instances'
    coco_dict['images'] = []
    coco_dict['categories'] = []
    coco_dict['annotations'] = []
    # make category sub-dataset json file
    coco_dict['categories'].append({'supercategory':'none', 'id':1, 'name': category})
    for label_index in label_indexes[category_id]:
        img_name = bbox_data[label_index][1]
        img = cv2.imread(img_path+img_name)
        height, width = img.shape[0], img.shape[1] # row, column
        image_dict = {'id': label_index, 'file_name':img_name,\
                      'width':width, 'height': height}
        coco_dict['images'].append(image_dict)
        xmin, ymin, xmax, ymax = map(int, list(bbox_data[label_index][-4:]))
        polygons = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
        area = (xmax-xmin)*(ymax-ymin)
        ann_dict = {'segmentation':[polygons], 'area':area, 'iscrowd':0, 'ignore':0,\
                    'image_id':label_index, 'bbox':[xmin, ymin, xmax-xmin, ymax-ymin],\
                    'category_id':1, 'id':label_index}
        coco_dict['annotations'].append(ann_dict)

    return coco_dict

def make_keypoints_json(img_path, ann_path):
    ann_data = pd.read_csv(ann_path).values
    # make keypoints dataset dict
    img_labels = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
    total_kpts=['neckline_left','neckline_right','center_front','shoulder_left','shoulder_right','armpit_left','armpit_right',
                'waistline_left','waistline_right','cuff_left_in','cuff_left_out','cuff_right_in','cuff_right_out',
                'top_hem_left','top_hem_right','waistband_left','waistband_right','hemline_left','hemline_right','crotch',
                'bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out']
    category_kpts_idx=[np.array([0,1,2,3,4,5,6,9,10,11,12,13,14]),
                       np.array([15,16,17,18]),
                       np.array([0,1,3,4,5,6,7,8,9,10,11,12,13,14]),
                       np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,17,18]),
                       np.array([15,16,19,20,21,22,23])]
    category_kpts=[total_kpts[:7]+total_kpts[9:15],
                   total_kpts[-9:-5],
                   total_kpts[:2]+total_kpts[3:15],
                   total_kpts[:13]+total_kpts[17:19],
                   total_kpts[-9:-7]+total_kpts[-5:]]
    '''
    blouse13:[u'neckline_left', u'neckline_right', u'center_front', u'shoulder_left', u'shoulder_right', u'armpit_left', u'armpit_right', 
            u'cuff_left_in', u'cuff_left_out', u'cuff_right_in', u'cuff_right_out', u'top_hem_left', u'top_hem_right']
    skirt4:['waistband_left','waistband_right','hemline_left','hemline_right']
    outwear14:['neckline_left','neckline_right','shoulder_left','shoulder_right','armpit_left','armpit_right','waistline_left','waistline_right',
            'cuff_left_in','cuff_left_out','cuff_right_in','cuff_right_out','top_hem_left','top_hem_right']
    dress15:[ 'neckline_left','neckline_right','center_front','shoulder_left','shoulder_right','armpit_left','armpit_right',
            'waistline_left','waistline_right','cuff_left_in','cuff_left_out','cuff_right_in','cuff_right_out','hemline_left','hemline_right']     
    trousers7:['waistband_left','waistband_right','crotch','bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out']
    '''
    coco_dicts=[]
    for i in range(len(img_labels)):
#         # add for bbox cropped images
#         if i!= 1:
#             continue
        
        category_id = np.where(ann_data==img_labels[i])[0]
        coco_dict = {}
        coco_dict['info'] = {'description': 'FashionAI Dataset', 'url':'https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100067.5678.1.510044a9T425c6&raceId=231648'}
        coco_dict['licenses'] = [{u'url': u'http://creativecommons.org/licenses/by-nc-nd/2.0/', u'id': 3, u'name': u'Attribution-NonCommercial-NoDerivs License'},]
        coco_dict['images'] = []
        coco_dict['categories'] = []
        coco_dict['annotations'] = []
        
        # make category sub-dataset json file
        coco_dict['categories'].append({'supercategory':'person', 'id':1, 'name': 'person', # coco_keypoints_json only accept person class name
                                        'keypoints':category_kpts[i]})
        print len(category_kpts_idx[i]),len(category_kpts[i])
        print category_kpts[i]
        for row_id in category_id:
            img_name = ann_data[row_id][0]
            print img_name.split('/')[-1]
            img = cv2.imread(img_path+img_name)
            height, width = img.shape[0], img.shape[1]
            image_dict = {'license':3, 'id': row_id, 'file_name':img_name,\
                      'width':width, 'height': height}
            coco_dict['images'].append(image_dict)
#             print ann_data[row_id][6+category_kpts_idx[i]]
            xmin, ymin, xmax, ymax = map(int, list(ann_data[row_id][2:6]))
            kpts = []
            for kpt_coord in ann_data[row_id][6+category_kpts_idx[i]]:
                kpt_coord = map(int,(kpt_coord.split('_')))
                if kpt_coord[-1]==-1:
                    kpt_coord = [0,0,0]
                else:
                    kpt_coord[-1]+=1
#                     # add for bbox cropped images
#                     kpt_coord[0]=kpt_coord[0]-xmin
#                     kpt_coord[1]=kpt_coord[1]-ymin
                
                kpts.extend(kpt_coord)            
            
#             # add for bbox cropped images
#             xmin, ymin, xmax, ymax = [0, 0, xmax-xmin, ymax-ymin]
#             assert (xmax-xmin)*(ymax-ymin) > 0, img_name+' area is negative'
    
            polygons = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
            area = (xmax-xmin)*(ymax-ymin)
            ann_dict = {'segmentation':[polygons], 'area':area, 'iscrowd':0, 'ignore':0,\
                        'image_id':row_id, 'bbox':[xmin, ymin, xmax-xmin, ymax-ymin],\
                        'category_id':1, 'id':row_id, 'num_keypoints':len(category_kpts_idx[i]),
                        'keypoints': kpts}  
            print kpts
            coco_dict['annotations'].append(ann_dict) 
        coco_dicts.append(coco_dict)
    print 'finished!'
    return coco_dicts
    
def make_bbox_imgs():
    """Make cropped images for training kpts detection model according bbox"""
    dataset_path = '/home/ygj/Software/Detectron/lib/datasets/data/FashionAI/train/'
#    img_path = dataset_path+'/Images/'+category
    ann_path = dataset_path+'Annotations/train_keypoint.csv'
    ann = pd.read_csv(ann_path).values[7158:14776]
    for record in ann:
        img_name = dataset_path+record[0]
        xmin,ymin,xmax,ymax = record[2:6]
        print(img_name)
        img = cv2.imread(img_name)
        cropped_img = img[ymin:ymax+1,xmin:xmax+1]
        cropped_img_name = dataset_path+'Images/bbox_cropped_imgs/'+record[0]
        cv2.imwrite(cropped_img_name, cropped_img)
    print('Finished!')
    
def merge_a_class(dict1, dict2, category_id):
    len1 = len(dict1['images'])
    for i in range(len(dict2['images'])):
        dict2['images'][i]['id'] = len1+i
        dict1['images'].append(dict2['images'][i])
        dict2['annotations'][i]['image_id']= len1+i
        dict2['annotations'][i]['id'] = len1+i
        dict2['annotations'][i]['category_id'] = category_id
        dict1['annotations'].append(dict2['annotations'][i])
    
def merge_classes(dict1, dict2=None, dict3=None, dict4=None, dict5=None):
    """Merge list value of dict with the same key"""
    merge_dic = {}
    merge_dic['type'] = 'instances'
    merge_dic['categories'] = []
    
#     # merge all five class
#     img_labels = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
#     for id in range(len(img_labels)):
#         merge_dic['categories'].append({'supercategory':'none', 'id': id+1, 'name': img_labels[id]})
#     assert dict1['categories'][0]['name']=='blouse'
#     merge_dic['images'] = copy.deepcopy(dict1['images'])
#     merge_dic['annotations'] = copy.deepcopy(dict1['annotations'])
#     # add items of other dicts and modify index
#     assert dict2['categories'][0]['name']=='skirt'
#     merge_a_dict(merge_dic, dict2, 2)    
#     assert dict3['categories'][0]['name']=='outwear'
#     merge_a_dict(merge_dic, dict3, 3)  
#     assert dict4['categories'][0]['name']=='dress'
#     merge_a_dict(merge_dic, dict4, 4)
#     assert dict5['categories'][0]['name']=='trousers'
#     merge_a_dict(merge_dic, dict5, 5)
    
    img_labels = ['skirt','trousers']
    for id in range(len(img_labels)):
        merge_dic['categories'].append({'supercategory':'none', 'id': id+1, 'name': img_labels[id]})
    assert dict1['categories'][0]['name']=='skirt'
    merge_dic['images'] = copy.deepcopy(dict1['images'])
    merge_dic['annotations'] = copy.deepcopy(dict1['annotations'])
    # add items of other dicts and modify index   
    assert dict2['categories'][0]['name']=='trousers'
    merge_a_class(merge_dic, dict2, 2) 
    
    return merge_dic

def test(file_name, idx):
    """Test make_coco_json function"""
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        for key in dataset:
            print key
#         print dataset['type']
        print len(dataset['images']), dataset['images'][idx]
        print len(dataset['categories']), dataset['categories'][0]
        print len(dataset['annotations']), dataset['annotations'][idx]

def merge_original_train_val():
    """merge original train/val dataset"""
    json_path = '/home/ygj/EclipseWorkspace/FashionAI_keypoint/'
    label_list = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
    for label in label_list:
        train_json = json_path+label+'_train.json'
        val_json = json_path+label+'_val.json'
        dict1 = json.load(open(train_json, 'r'))
        dict2 = json.load(open(val_json, 'r'))
        for i in range(len(dict2['images'])):
            dict1['images'].append(dict2['images'][i])
            dict1['annotations'].append(dict2['annotations'][i])
        file_name = label+'_train_new.json'
        json.dump(dict1, open(file_name, 'w'))
        test(file_name, 0)

def split_kpts_dataset():
    """split  4/5 of xxx_keypoints_train.json
        as training set and the rest 1/5 of xxx_keypoints_train.json as val set"""
    json_path = '/home/ygj/EclipseWorkspace/FashionAI_keypoint/keypoint_json/'
    label_list = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
    for label in label_list:
        train_json = json_path+label+'_keypoints_trainval.json'
        train_dict = json.load(open(train_json, 'r'))
        split_idx = int(round(0.8*(len(train_dict['images']))))      
         
        train_new = {}
        train_new['info'] = train_dict['info']
        train_new['licenses'] = train_dict['licenses']
        train_new['categories'] = train_dict['categories']
        train_new['images'] = train_dict['images'][:split_idx]
        train_new['annotations'] = train_dict['annotations'][:split_idx]
        train_new_path = json_path+label+'_keypoints_train.json'
        json.dump(train_new, open(train_new_path, 'w'))
         
        train_dict['images'] = train_dict['images'][split_idx:]
        train_dict['annotations'] = train_dict['annotations'][split_idx:]
        val_new_path = json_path+label+'_keypoints_val.json'
        json.dump(train_dict, open(val_new_path, 'w'))
        test(train_new_path, 0)
        test(val_new_path, 0)
       
if __name__=='__main__':
    img_path ='/home/ygj/Software/Detectron/lib/datasets/data/FashionAI/warm_up/'
    bbox_path = 'val_bounding_box.csv'

#     # make bbox cropped skirt images
#     make_bbox_imgs()
#     ann_path = '/home/ygj/Software/Detectron/lib/datasets/data/FashionAI/train/Annotations/train_keypoint.csv'
#     file_name = '/home/ygj/EclipseWorkspace/FashionAI_keypoint/keypoint_json/skirt_cropped_keypoints_train.json'
#     kpt_coco_dict = make_keypoints_json(img_path, ann_path)[0]
#     json.dump(kpt_coco_dict, open(file_name, 'w'))
#     test(file_name,0)

#    split_kpts_dataset()

#     # make keypoints dataset
#     ann_path = '/home/ygj/Software/Detectron/lib/datasets/data/FashionAI/warm_up/Annotations/annotations_keypoint.csv'
#     kpt_coco_dicts=make_keypoints_json(img_path, ann_path)
#     label_list = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
#     for i in range(5):
#         file_name = '/home/ygj/EclipseWorkspace/FashionAI_keypoint/keypoint_json/'+label_list[i]+'_keypoints_warm_up.json'
#         json.dump(kpt_coco_dicts[i], open(file_name, 'w'))
#         test(file_name, 0)
    
    file_name = '/home/ygj/Software/Detectron/lib/datasets/data/FashionAI/train/Annotations/trousers_keypoints_train.json'
    test(file_name,0) 
       
#     # make warm_up validation dataset
#     label_list = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
#     for label in label_list:
#         warm_up = make_warm_up_json(img_path, bbox_path, label)
#         file_name = label+'_val_new.json'
#         json.dump(warm_up, open(file_name, 'w'))
#         test(file_name, 0)
    
#     # make a kind of class train/val datase
#     train, val = make_coco_json(img_path, bbox_path, 'trousers')
#     json.dump(train, open('trousers_train.json', 'w'))
#     json.dump(val, open('trousers_val.json', 'w'))
#     test('trousers_train.json',0)
#     test('trousers_val.json',0)
    
    # make total train/val dataset
#     blouse_train, blouse_val = make_coco_json(img_path, bbox_path, 'blouse')
#     skirt_train, skirt_val = make_coco_json(img_path, bbox_path, 'skirt')
#     outwear_train, outwear_val = make_coco_json(img_path, bbox_path, 'outwear')
#     dress_train, dress_val = make_coco_json(img_path, bbox_path, 'dress')
#     trousers_train, trousers_val = make_coco_json(img_path, bbox_path, 'trousers')
    
#     total_train = merge_dict(blouse_train, skirt_train, outwear_train, dress_train, trousers_train)
#     total_val = merge_dict(blouse_val, skirt_val, outwear_val, dress_val, trousers_val)   
#     json.dump(total_train, open('FashionAI_train.json', 'w'))
#     json.dump(total_val, open('FashionAI_val.json', 'w'))
#     test('FashionAI_train.json')
#     test('FashionAI_val.json')

    # make skirt_trousers dataset json file
#     skirt_train = json.load(open('skirt_train.json'))
#     skirt_val = json.load(open('skirt_val.json'))
#     trousers_train = json.load(open('trousers_train.json'))
#     trousers_val = json.load(open('trousers_val.json'))
#     skirt_trousers_train = merge_dict(skirt_train, trousers_train)
#     skirt_trousers_val = merge_dict(skirt_val, trousers_val)
#     
#     json.dump(skirt_trousers_train, open('skirt_trousers_train.json', 'w'))
#     json.dump(skirt_trousers_val, open('skirt_trousers_val.json', 'w'))
#     test('skirt_trousers_train.json')
#     test('skirt_trousers_val.json')
    