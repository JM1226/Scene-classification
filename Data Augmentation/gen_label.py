import json
import os
aug=[]
namelist=os.listdir('train_aug')
num=0
for name in namelist:
    num+=1
    print(num)
    label=name.split('#')[0]
    aug.append({'image_id':name,'label_id':label})
fp=open('aug.json','w')
json.dump(aug,fp,indent=4)
print("Done")