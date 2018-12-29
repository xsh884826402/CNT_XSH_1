import xpinyin
#程序说明
#   Sort_by_Pinyin按照拼音对输入的二维list进行排序
#   输入：二维病历列表，例如[['a','b'],['c','d']]
#   功能：将汉字转换成拼音，对拼音排序，再将拼音转换成汉字
#   输出：按照拼音排过序的二维列表
def Sort_by_Pinyin(list_2_dim):

    dict_HanZi_to_Pinyin={}
    dict_Pinyin_to_Hanzi ={}

    Sorted_list_2_dim = []

    Pinyin_transfer = xpinyin.Pinyin()
    for list_1d in list_2_dim:
        for item in list_1d:
            temp_item =Pinyin_transfer.get_pinyin(item)
            dict_HanZi_to_Pinyin[item] = temp_item
            dict_Pinyin_to_Hanzi[temp_item] = item
    for list_1d in list_2_dim:
        temp_list_1d =[]
        temp_list_1d_1 = []
        for item in list_1d:
            temp_list_1d.append(dict_HanZi_to_Pinyin[item])
        temp_list_1d = sorted(temp_list_1d,)
        # print('temp_list_1d',temp_list_1d)
        for item in temp_list_1d:
            temp_list_1d_1.append(dict_Pinyin_to_Hanzi[item])
        Sorted_list_2_dim.append(temp_list_1d_1)
    return Sorted_list_2_dim
if __name__ =="__main__":
    test_list = [['小便不利', '小便涩痛', '尿急', '腰痛', '便溏', '阴痒', '舌红', '舌苔黄', '舌苔腻'], ['带下色黄', '带下腥臭', '带下量多', '大便黏腻', '舌尖红', '舌苔厚', '舌苔腻', '阴痒']]
    print('before result', sorted(test_list))
    result = Sort_by_Pinyin(test_list)
    print('result', result)
