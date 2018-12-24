<<<<<<< HEAD
import xlwt
def list_to_xls(input_list, file_name):
    data = xlwt.Workbook()
    tabel = data.add_sheet("Result")
    i = 1
    for item in input_list:
        item_1, item_2 = item
        tabel.write(i,0,item_1)
        tabel.write(i,1,item_2)
        i = i+1
    data.save(file_name)
def list_to_xls2(input_list, file_name):
    data = xlwt.Workbook()
    tabel = data.add_sheet("Result")
    i = 1
    for item in input_list:
        item_1, item_2, item_3 = item
        str_temp = ""
        str_temp_1 = ""
        for j in range(len(item_2)):
            str_temp = str_temp + "".join(item_2[j])
            str_temp_1 = str_temp_1 + "".join(item_3[j])
        print('str', str_temp, str_temp_1)
        tabel.write(i,0,item_1)
        tabel.write(i,1,str_temp)
        tabel.write(i,2,str_temp_1)
        i = i+1
    data.save(file_name)
def list_to_xls3(input_list_1,input_list_2,input_list_3, file_name):
    #增加了病历数据
    data = xlwt.Workbook()
    tabel = data.add_sheet("Result")
    i = 1
    for k in range(len(input_list_1)):
    # for item in input_list_1:
        item_1, item_2, item_3 = input_list_1[k]
        item_4 = input_list_2[k]
        item_5 = input_list_3[k]
        str_temp = ""
        str_temp_1 = ""
        for j in range(len(item_2)):
            str_temp = str_temp + "".join(item_2[j])
            str_temp_1 = str_temp_1 + "".join(item_3[j])
        # print('str', str_temp, str_temp_1)
        print()
        tabel.write(i,0,item_1)
        tabel.write(i,1,str_temp)
        tabel.write(i,2,str_temp_1)
        tabel.write(i,3,item_4)
        tabel.write(i,4,item_5)
        i = i+1
    data.save(file_name)
def list_to_xls4(input_list_1, input_list_2, input_list_3, file_name):
        # 将各个辨证分别输出
        data = xlwt.Workbook()
        table = data.add_sheet("Result")
        i = 1
        for k in range(len(input_list_1)):
            # for item in input_list_1:
            item_1, item_2, item_3 = input_list_1[k]
            item_4 = input_list_2[k]
            item_5 = input_list_3[k]
            str_temp = ""
            str_temp_1 = ""
            for j in range(len(item_2)):
                str_temp = str_temp + "".join(item_2[j])
                str_temp_1 = str_temp_1 + "".join(item_3[j])
            # print('str', str_temp, str_temp_1)

            table.write(i, 0, item_1)
            table.write(i, 1, item_2[0])
            table.write(i, 2, item_2[1])
            table.write(i, 3, item_2[2])
            table.write(i, 4,item_2[3])
            table.write(i, 5, item_2[4])
            table.write(i, 6, str_temp_1)
            table.write(i, 7, item_4)
            i = i + 1
        data.save(file_name)

=======
import xlwt
def list_to_xls(input_list, file_name):
    data = xlwt.Workbook()
    tabel = data.add_sheet("Result")
    i = 1
    for item in input_list:
        item_1, item_2 = item
        tabel.write(i,0,item_1)
        tabel.write(i,1,item_2)
        i = i+1
    data.save(file_name)
def list_to_xls2(input_list, file_name):
    data = xlwt.Workbook()
    tabel = data.add_sheet("Result")
    i = 1
    for item in input_list:
        item_1, item_2, item_3 = item
        str_temp = ""
        str_temp_1 = ""
        for j in range(len(item_2)):
            str_temp = str_temp + "".join(item_2[j])
            str_temp_1 = str_temp_1 + "".join(item_3[j])
        print('str', str_temp, str_temp_1)
        tabel.write(i,0,item_1)
        tabel.write(i,1,str_temp)
        tabel.write(i,2,str_temp_1)
        i = i+1
    data.save(file_name)
def list_to_xls3(input_list_1,input_list_2,input_list_3, file_name):
    #增加了病历数据
    data = xlwt.Workbook()
    tabel = data.add_sheet("Result")
    i = 1
    for k in range(len(input_list_1)):
    # for item in input_list_1:
        item_1, item_2, item_3 = input_list_1[k]
        item_4 = input_list_2[k]
        item_5 = input_list_3[k]
        str_temp = ""
        str_temp_1 = ""
        for j in range(len(item_2)):
            str_temp = str_temp + "".join(item_2[j])
            str_temp_1 = str_temp_1 + "".join(item_3[j])
        # print('str', str_temp, str_temp_1)
        print()
        tabel.write(i,0,item_1)
        tabel.write(i,1,str_temp)
        tabel.write(i,2,str_temp_1)
        tabel.write(i,3,item_4)
        tabel.write(i,4,item_5)
        i = i+1
    data.save(file_name)
def list_to_xls4(input_list_1, input_list_2, input_list_3, file_name):
        # 将各个辨证分别输出
        data = xlwt.Workbook()
        table = data.add_sheet("Result")
        i = 1
        for k in range(len(input_list_1)):
            # for item in input_list_1:
            item_1, item_2, item_3 = input_list_1[k]
            item_4 = input_list_2[k]
            item_5 = input_list_3[k]
            str_temp = ""
            str_temp_1 = ""
            for j in range(len(item_2)):
                str_temp = str_temp + "".join(item_2[j])
                str_temp_1 = str_temp_1 + "".join(item_3[j])
            # print('str', str_temp, str_temp_1)

            table.write(i, 0, item_1)
            table.write(i, 1, item_2[0])
            table.write(i, 2, item_2[1])
            table.write(i, 3, item_2[2])
            table.write(i, 4,item_2[3])
            table.write(i, 5, item_2[4])
            table.write(i, 6, str_temp_1)
            table.write(i, 7, item_4)
            i = i + 1
        data.save(file_name)

>>>>>>> 0788edf81b2bbb8ef6e66d48738a23171b113672
