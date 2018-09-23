import pandas as pd
import cv2
import numpy as np
import pdb
import os

def show_img_and_points(img_path,train_data,predict_data=None):
    for i,labels in enumerate(train_data):
        img_name,img_label = labels[0],labels[1]
        img = cv2.imread(img_path+img_name)
        cv2.rectangle(img, (labels[2],labels[3]),(labels[4],labels[5]),(0,255,0),2)
        for label_axis in labels[6:]: # original is labels[2:]
            label_axis = label_axis.split('_')
            label_axis = map(float, label_axis)
            if label_axis[0] > 0 and label_axis[1] > 0:
                cv2.circle(img, (int(label_axis[0]),int(label_axis[1])), 2, (0,255,0),-1)
        # show predict points
        if predict_data is not None:
            cv2.rectangle(img, (predict_data[i][3],predict_data[i][4]),(predict_data[i][5],predict_data[i][6]),(0,0,255),2)
            for label_axis in predict_data[i][7:]:
                label_axis = label_axis.split('_')
                label_axis = map(float, label_axis)
                if label_axis[0] > 0 and label_axis[1] > 0:
                    cv2.circle(img, (int(label_axis[0]),int(label_axis[1])), 2, (0,0,255),-1)
        cv2.imshow('test',img)
        print(img_name)
        cv2.waitKey(0)
        
def show_test_points():
    img_path = '/media/ygj/00030DB30006F338/ygj/dataset/FashionAIdataset/FashionAI/test/'
    predict_path = './output/mask_rcnn_skirt_test_infer2.csv'
    predict_data = pd.read_csv(predict_path).values
    for i,labels in enumerate(predict_data):
        img_name,img_label = labels[1],labels[2]
        img = cv2.imread(img_path+img_name)
        cv2.rectangle(img, (labels[3],labels[4]),(labels[5],labels[6]),(0,255,0),2)
        for label_axis in predict_data[i][7:]:
            label_axis = label_axis.split('_')
            label_axis = map(float, label_axis)
            if label_axis[0] > 0 and label_axis[1] > 0:
                cv2.circle(img, (int(label_axis[0]),int(label_axis[1])), 2, (0,0,255),-1)
        cv2.imshow('test',img)
        cv2.waitKey(0)
    

    
def show_img_bbox(img_path, data_path, bbox_path):
    train_data = pd.read_csv(data_path).values[:1977]
    bbox_data = pd.read_csv(bbox_path).values
#    pdb.set_trace()
    data = np.concatenate((train_data, bbox_data[:,-4:]), axis = 1)
    print bbox_data.shape[1], data.shape[1]
    img_labels = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
    for name in img_labels:
        name_index = np.where(bbox_data==name)[0]
        print name, len(name_index)
    for labels in data:
        img_name,img_label, label_axises, bbox = labels[0],labels[1], labels[2:-4], map(int, labels[-4:])
#        img_name,img_label, bbox = labels[1],labels[2], map(int, labels[-4:])
        img = cv2.imread(img_path+img_name)
        green = (0, 255, 0)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), green, 2)
        for label_axis in label_axises:
            label_axis = label_axis.split('_')
            label_axis = map(float, label_axis)
            if label_axis[0] > 0 and label_axis[1] > 0:
                cv2.circle(img, (int(label_axis[0]),int(label_axis[1])), 2, (0,0,255),-1)
        print img_name
        cv2.imshow(img_label, img)
        cv2.waitKey(0) 
        
def load_data(img_path,data_path):
    train_data = pd.read_csv(data_path).values
    print len(train_data)
    show_img_and_points(img_path,train_data)
#    get_bounding_box(img_path,train_data)
    print'finished'
    
def plot_point_of_img(img_path):
    im = cv2.imread(img_path)
    im_shape = im.shape # [height, width, channel]
    cv2.circle(im, (10, 100), 2, (0, 0, 255), -1) # (10, 100):(col_index, row_index)
    print im_shape
    cv2.imshow('point(10, 100) of image', im)
    cv2.waitKey(0)
        
def plot_learning_curves():
    log_path = './output/blouse-30000iter/accuracy_loss_log.txt'
    #log_path = '/home/ygj/Software/Detectron/demo/output/D-169-FPN_07_12/70000_concat_out/accuracy_loss_log.txt'
    accuracy_list = []
    loss_list = []
    with open(log_path) as f:
        for line in f.readlines():
            accuracy, loss = map(float, line.strip().split(' '))
            accuracy_list.append(accuracy)
            loss_list.append(loss)
        
    plt.plot(accuracy_list, 'b')
    plt.plot(loss_list, 'r')
    plt.xlim([0, 300])
    plt.xlabel('iter/100')
    plt.ylabel('accuracy/loss')
    plt.title('training accuracy/loss curves')
    # plt.legend('Accuracy_cls', 'Loss', loc='upper right') 
    plt.show()
   
if __name__ == '__main__':
#     img_path = '/media/ygj/00030DB30006F338/ygj/dataset/FashionAIdataset/FashionAI/train/'
#     data_path = img_path + 'Annotations/train_keypoint.csv'
# #    load_data(img_path,data_path)
#     train_data = pd.read_csv(data_path).values[14776:14876]
#     predict_path = './output/train_R-50_outwear.csv'
#     predict_data = pd.read_csv(predict_path).values
#     show_img_and_points(img_path, train_data, predict_data)
    
    show_test_points()

#     img_path = '/media/ygj/00030DB30006F338/ygj/dataset/FashionAIdataset/FashionAI/test/'
#     data_path = img_path+'test.csv'
#     bbox_path = '/home/ygj/EclipseWorkspace/FashionAI_keypoint/output/test_bbox_R-50_blouse.csv'
#     show_img_bbox(img_path, data_path, bbox_path)


#    plot_point_of_img('/home/ygj/Software/Detectron/lib/datasets/data/FashionAI/test/Images/trousers/dfb5ad535bb37a605c643a36263513f4.jpg')
