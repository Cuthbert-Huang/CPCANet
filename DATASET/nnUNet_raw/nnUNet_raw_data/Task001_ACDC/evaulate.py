import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
from sklearn.neighbors import KDTree
from scipy import ndimage
import argparse


def read_nii(path):
    itk_img=sitk.ReadImage(path)
    return sitk.GetArrayFromImage(itk_img)

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def process_label(label):
    rv = label == 1
    myo = label == 2
    lv = label == 3
    
    return rv,myo,lv


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
   
    Dice_rv=[]
    Dice_myo=[]
    Dice_lv=[]
    
    hd_rv=[]
    hd_myo=[]
    hd_lv=[]
    
    file=path + 'inferTs/'+fold
    vol_file= path + 'inferTs/'+fold +'/complete_vol'

    if not os.path.exists(file):
        os.makedirs(file)
    if not os.path.exists(vol_file):
        os.makedirs(vol_file)
        
    fw = open(vol_file+'/dice_pre.txt', 'w')
    
    for label_path in label_list:
        infer_path=[i for i in infer_list if label_path.split('/')[-1][:-7] in i]
        print(infer_path)
        print(label_path)
        label=read_nii(label_path)
        
        assert label.shape[0]==len(infer_path)
        infer=np.zeros_like(label)
        for i in infer_path:
            idx=i.split('/')[-1].split('_')[-1][:-7]
            infer[int(idx)]=read_nii(i)
            
        img=sitk.GetImageFromArray(infer)
        sitk.WriteImage(img,os.path.join(vol_file,label_path.split('/')[-1]))
        
        label_rv,label_myo,label_lv=process_label(label)
        infer_rv,infer_myo,infer_lv=process_label(infer)
        
        Dice_rv.append(dice(infer_rv,label_rv))
        Dice_myo.append(dice(infer_myo,label_myo))
        Dice_lv.append(dice(infer_lv,label_lv))
        
        hd_rv.append(hd(infer_rv,label_rv))
        hd_myo.append(hd(infer_myo,label_myo))
        hd_lv.append(hd(infer_lv,label_lv))
        
        fw.write('*'*20+'\n',)
        fw.write(label_path.split('/')[-1]+'\n')
        fw.write('hd_rv: {:.4f}\n'.format(hd_rv[-1]))
        fw.write('hd_myo: {:.4f}\n'.format(hd_myo[-1]))
        fw.write('hd_lv: {:.4f}\n'.format(hd_lv[-1]))
        fw.write('*'*20+'\n',)
        fw.write(label_path.split('/')[-1]+'\n')
        fw.write('Dice_rv: {:.4f}\n'.format(Dice_rv[-1]))
        fw.write('Dice_myo: {:.4f}\n'.format(Dice_myo[-1]))
        fw.write('Dice_lv: {:.4f}\n'.format(Dice_lv[-1]))
        fw.write('hd_rv: {:.4f}\n'.format(hd_rv[-1]))
        fw.write('hd_myo: {:.4f}\n'.format(hd_myo[-1]))
        fw.write('hd_lv: {:.4f}\n'.format(hd_lv[-1]))
        fw.write('*'*20+'\n')
        

    
    fw.write('*'*20+'\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_rv'+str(np.mean(Dice_rv))+'\n')
    fw.write('Dice_myo'+str(np.mean(Dice_myo))+'\n')
    fw.write('Dice_lv'+str(np.mean(Dice_lv))+'\n')  
    fw.write('Mean_HD\n')
    fw.write('HD_rv'+str(np.mean(hd_rv))+'\n')
    fw.write('HD_myo'+str(np.mean(hd_myo))+'\n')
    fw.write('HD_lv'+str(np.mean(hd_lv))+'\n')    
    fw.write('*'*20+'\n')
    
    dsc=[]
    dsc.append(np.mean(Dice_rv))
    dsc.append(np.mean(Dice_myo))
    dsc.append(np.mean(Dice_lv))
    avg_hd=[]
    avg_hd.append(np.mean(hd_rv))
    avg_hd.append(np.mean(hd_myo))
    avg_hd.append(np.mean(hd_lv))

    fw.write('DSC:'+str(np.mean(dsc))+'\n')
    fw.write('HD:'+str(np.mean(avg_hd))+'\n')

    print('done')
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", help="fold name")
    args = parser.parse_args()
    fold=args.fold
    test(fold)