import os
import pickle
def data_build(docpath):
    text = []
    labels = []
    labels_dict = {}
    i=0
    for dir in os.listdir(docpath):
        print(dir)
        for dir_1 in os.listdir(os.path.join(docpath,dir)):
            file1 = open(os.path.join(docpath, dir, dir_1), 'r', encoding='utf-8')
            text_temp = file1.readline().strip().split()
            text.append(text_temp)
            file1.close()
            labels.append(i)
        labels_dict[i] = dir
        i += 1
    return text, labels
def data_build_label(docpath):
    text = []
    labels = []
    labels_dict = {}
    i=0
    for dir in os.listdir(docpath):
        # print(dir)
        for dir_1 in os.listdir(os.path.join(docpath,dir)):
            file1 = open(os.path.join(docpath, dir, dir_1), 'r', encoding='utf-8')
            text_temp = file1.readline().strip().split()
            text.append(text_temp)
            file1.close()
            labels.append(dir)
        # labels_dict[i] = dir
        # i += 1
    return text, labels
def data_build_dict(docpath):
    text = []
    labels = []
    labels_dict = {}
    i=0
    labels_dict_dir=open('./labels_dict', 'wb')
    for dir in os.listdir(docpath):
        for dir_1 in os.listdir(os.path.join(docpath,dir)):
            file1 = open(os.path.join(docpath, dir, dir_1), 'r', encoding='utf-8')
            text_temp = file1.readline().strip().split()
            text.append(text_temp)
            file1.close()
            labels.append(i)
        labels_dict[i] = dir
        i += 1
    pickle.dump(labels_dict, labels_dict_dir)
def data_build_1(docpath):
    text = []
    labels = []
    labels_dict = {}
    i=0
    for dir in os.listdir(docpath):
        file = open(os.path.join(docpath, dir), 'r', encoding='utf-8')
        text_temp = file.readline().strip().split()
        print('text', text_temp)
        text.append(text_temp)
        file.close()
    # print(text)
    return text
if __name__ =='__main__':
    data_build_1('./data/data15/test_1')