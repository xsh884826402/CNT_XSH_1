#kmeans  先根据data/辨证分词20181003.xlsx中的文本向量化（通过multilabel向量化，将五个辨证concatenate，然后输入kmeans
import sklearn.preprocessing as pre
import pickle
import numpy as np
from sklearn.cluster import KMeans
#Load label_transfer_dict.pkl
with open('./data/label_transfer_dict.pkl', 'rb') as f:
    label_transfer_dict = pickle.load(f)
#label_transfer_keys represent 40 category,maybe used for Kmeans label
label_transfer_keys = label_transfer_dict.keys()
#dict_for_kmeans key 是各个类别的标签，value是一个28维的数组
dict_for_kmeans = {}
#five MultiLabelBinarizer
labels_1 = ['心肝脾肺肾胆胃']
labels_2 = ['气','血','湿','痰','泛','水','瘀']
labels_3 = ['阴阳表里虚实寒热']
labels_4 = ['卫','气','血']
labels_5 = ['上','中','下']

mlb_1 = pre.MultiLabelBinarizer()
mlb_2 = pre.MultiLabelBinarizer()
mlb_3 = pre.MultiLabelBinarizer()
mlb_4 = pre.MultiLabelBinarizer()
mlb_5 = pre.MultiLabelBinarizer()

mlb_1.fit(labels_1)
mlb_2.fit(labels_2)
mlb_3.fit(labels_3)
mlb_4.fit(labels_4)
mlb_5.fit(labels_5)

for key in label_transfer_keys:
    temp_zangfu_str = label_transfer_dict[key][3]
    if temp_zangfu_str != temp_zangfu_str:
        temp_zangfu_str = ''
    temp_zangfu_list = []
    temp_zangfu_list.append(temp_zangfu_str)
    temp_zangfu_vector = mlb_1.transform(temp_zangfu_list)
    # print('shape', np.shape(temp_zangfu_vector),temp_zangfu_vector,)
    temp_array =temp_zangfu_vector
    # print('key ZangFu',key,temp_zangfu_vector ,temp_array,end='\r')

    temp_qixuejinye_str = label_transfer_dict[key][4]
    if temp_qixuejinye_str != temp_qixuejinye_str:
        temp_qixuejinye_str = ''
    temp_qixuejinye_list = []
    temp_qixuejinye_list.append(temp_qixuejinye_str)
    temp_qixuejinye_vector = mlb_2.transform(temp_qixuejinye_list)
    temp_array = np.concatenate((temp_array, temp_qixuejinye_vector),axis=1)
    # print('key qixuejinye', temp_array)

    temp_bagang_str = label_transfer_dict[key][5]
    if temp_bagang_str != temp_bagang_str:
        temp_bagang_str = ''
    temp_bagang_list = []
    temp_bagang_list.append(temp_bagang_str)
    temp_bagang_vector = mlb_3.transform(temp_bagang_list)
    temp_array = np.concatenate((temp_array, temp_bagang_vector), axis=1)

    temp_weiqiyingxue_str = label_transfer_dict[key][6]
    if temp_weiqiyingxue_str != temp_weiqiyingxue_str:
        temp_weiqiyingxue_str = ''
    temp_weiqiyingxue_list= []
    temp_weiqiyingxue_list.append(temp_weiqiyingxue_str)
    temp_weiqiyingxue_vector = mlb_4.transform(temp_weiqiyingxue_list)
    temp_array = np.concatenate((temp_array, temp_weiqiyingxue_vector), axis =1)

    temp_sanjiao_str = label_transfer_dict[key][7]
    if temp_sanjiao_str != temp_sanjiao_str:
        temp_sanjiao_str = ''
    temp_sanjiao_list = []
    temp_sanjiao_list.append(temp_sanjiao_str)
    temp_sanjiao_vector = mlb_5.transform(temp_sanjiao_list)
    temp_array = np.concatenate((temp_array, temp_sanjiao_vector), axis= 1)
    # print('temp_array',temp_array,temp_array.shape, temp_array.size)
    # print('key temp',key ,temp_array)
    dict_for_kmeans[key] = temp_array

# array_for_kmeans = dict_for_kmeans['肺经蕴热']
# print(array_for_kmeans)
# array_for_kmeans = np.empty(shape=(1,28))
flag = 1
print(len(list(dict_for_kmeans.keys())))
print(len(dict_for_kmeans.values()))
for key in dict_for_kmeans.keys():
    if(flag ==1):
        array_for_kmeans = dict_for_kmeans[key]
        flag =0
        continue
    array_for_kmeans = np.concatenate((array_for_kmeans,dict_for_kmeans[key]), axis=0)
kmeans = KMeans(n_clusters=30).fit(array_for_kmeans)
y = kmeans.predict(array_for_kmeans)
print('Y', y)
print(label_transfer_keys)

