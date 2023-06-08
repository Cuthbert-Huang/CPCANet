import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
from sklearn.neighbors import KDTree
from scipy import ndimage
import argparse
from medpy.metric.binary import jc

def read_nii(path):
    itk_img=sitk.ReadImage(path)
    return sitk.GetArrayFromImage(itk_img)

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def process_label(label):
    les = label == 1
   
    
    return les


def hd(pred,gt):

    if pred.sum() > 0 and gt.sum()>0:
        hd95 = binary.hd95(pred, gt)
        print(hd95)
        return  hd95
    else:
        return 0






def test(fold):
    path='./'
    label_list=sorted(glob.glob(os.path.join(path,'labelsTs','*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(path,'inferTs',fold,'*nii.gz')))
    print("loading success...")
    print(label_list)
    print(infer_list)
    Dice=[]
   
    iou=[]

    
    file=path + 'inferTs/'+fold
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file+'/dice_pre.txt', 'w')
    
    for label_path,infer_path in zip(label_list,infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label= read_nii(label_path)
        infer= read_nii(infer_path)
       
        
        Dice.append(dice(infer,label))      
        iou.append(jc(infer, label))
        
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('iou: {:.4f}\n'.format(iou[-1]))
        fw.write('Dice: {:.4f}\n'.format(Dice[-1]))
        fw.write('*'*20+'\n')


    fw.write('DSC:'+str(np.mean(Dice))+'\n')
    fw.write('avg_iou:'+str(np.mean(iou))+'\n')
    
    print('done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", help="fold name")
    args = parser.parse_args()
    fold=args.fold
    test(fold)
