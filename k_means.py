#kmeans  先根据data/辨证分词20181003.xlsx中的文本向量化（通过multilabel向量化，将五个辨证concatenate，然后输入kmeans
import sklearn.preprocessing as pre
import pickle
#Load label_transfer_dict.pkl
with open('./data/label_transfer_dict.pkl', 'rb') as f:
    label_transfer_dict = pickle.load(f)
#label_transfer_keys represent 40 category,maybe used for Kmeans label
label_transfer_keys = label_transfer_dict.keys()
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
    temp_list = []

    temp_zangfu_str = label_transfer_dict[key][3]
    if temp_zangfu_str != temp_zangfu_str:
        temp_zangfu_str = ''
    temp_zangfu_vector = mlb_1.transform(temp_zangfu_str)
    temp_list.append(temp_zangfu_vector)

    temp_qixuejinye_str = label_transfer_dict[key][4]
    if temp_qixuejinye_str != temp_qixuejinye_str:
        temp_qixuejinye_str = ''
    temp_qixuejinye_vector = mlb_2.transform(temp_qixuejinye_str)
    temp_list.append(temp_qixuejinye_vector)

    temp_bagang_str = label_transfer_dict[key][5]
    if temp_bagang_str != temp_bagang_str:
        temp_bagang_str = ''
    temp_bagang_vector = mlb_3.transform(temp_bagang_str)
    temp_list.append(temp_bagang_vector)

    temp_weiqiyingxue_str = label_transfer_dict[key][6]
    if temp_weiqiyingxue_str != temp_weiqiyingxue_str:
        temp_weiqiyingxue_str = ''
    temp_weiqiyingxue_vector = mlb_4.transform(temp_weiqiyingxue_str)
    temp_list.append(temp_weiqiyingxue_vector)

    temp_sanjiao_str = label_transfer_dict[key][7]
    if temp_sanjiao_str != temp_sanjiao_str:
        temp_sanjiao_str = ''
    temp_sanjiao_vector = mlb_5.transform(temp_sanjiao_str)
    temp_list.append(temp_sanjiao_vector)
    print('key temp',key ,temp_list)
    dict_for_kmeans[key] = temp_list



# labels = ['心肝脾肺肾胆胃']
#     mlb = MultiLabelBinarizer()
#     mlb.fit(labels)
#     with open('./data/label_transfer_dict.pkl', 'rb') as f:
#         label_transfer_dict = pickle.load(f)
#
#     y_temp = []
#     count = 0
#     for item in y_train:
#             # print(item)
#             temp = label_transfer_dict[item][3]
#             if temp != temp:
#                 y_temp.append('')
#             else:
#                 y_temp.append(label_transfer_dict[item][3])
#     print(y_temp)
#     print(mlb.classes_)
#     y_multilabel = mlb.transform(y_temp)
#     print(y_multilabel)
#     print('type',type(y_multilabel))
#     print('shape', np.shape(y_multilabel))
#     print(mlb.inverse_transform(y_multilabel))
#     return y_multilabel