from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import preprocessing
# a = ['小便不利', '小便涩痛', '尿急', '腰痛', '便溏', '阴痒', '舌红', '舌苔黄', '舌苔腻','啊','吧','从']
# b = '我就是'
# b.encode('gbk')
#
# print(sorted(a))
# for item in a:
#     item.encode('gbk')
#
# print(sorted(a))
# print(b)
a = np.zeros([2,3])
a[0] = [1,2,3]
print(a)
