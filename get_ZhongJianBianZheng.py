#为方便处理，将血瘀处理成瘀
import pandas as pd
import pickle
#读取中间辨证那张excell表，将辨证与中间辨证以词典的形式存储，存储到label_transfer_dict
def get_dict_label(filename = './data/辩证分词20181003.xlsx'):
    df = pd.read_excel(filename)
    list_label = list(df.keys())
    dict_label = {}
    # print(type(df.ix[0,'中医辨证名称']),df.ix[0,'中医辨证名称'])
    # print(type(list_label), list_label)
    for i in range(40):
        key = df.ix[i, '中医辨证名称']
        list1 = []
        for name in list_label:
            list1.append(df.ix[i, name])

        dict_label[key] = list1
    return dict_label

if __name__ =="__main__":
    dict = get_dict_label('./data/辩证分词20181003.xlsx')
    with open('./data/label_transfer_dict.pkl','wb' ) as f:
        pickle.dump(dict, f)
    print('dict',dict)
    print('store suceess')