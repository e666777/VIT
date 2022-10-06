import pickle


with open('ViT_no_aug.txt','rb') as file_pi:
    history_no_aug=pickle.load(file_pi)
with open('ViT_aug.txt','rb') as file_pi:
    history_aug=pickle.load(file_pi)    
with open('ViT_mixup.txt','rb') as file_pi:
    history_mixup=pickle.load(file_pi)   
with open('ViT_cutmix.txt','rb') as file_pi:
    history_cutmix=pickle.load(file_pi)    


import matplotlib.pyplot as plt
plt.plot(history_no_aug['val_Accuracy'])
plt.plot(history_aug['val_Accuracy'])
plt.plot(history_mixup['val_Accuracy'])
plt.plot(history_cutmix['val_Accuracy'])
plt.title('Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['no_aug', 'aug', 'mixup', 'cutmix'], loc='lower right')
plt.show()


import matplotlib.pyplot as plt
plt.plot(history_no_aug['val_auc'])
plt.plot(history_aug['val_auc'])
plt.plot(history_mixup['val_auc'])
plt.plot(history_cutmix['val_auc'])
plt.title('Validation auc')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['no_aug', 'aug', 'mixup', 'cutmix'], loc='lower right')
plt.show()