<<<<<<< HEAD
def getText(datapath):
    f = open(datapath, 'r', encoding='utf-8')
    lists = f.readlines()
    # print(lists)
    text = []
    result = []
    for list in lists:
        if list != '\n':
            if list[-1] == '\n':
                list = list[0:-1]
            if list[-1]==' ':
                list = list[0:-1]
            text.append(list)

    return text
if __name__ =='__main__':
    text = getText('./data/TestHe.txt')
    print(text)
=======
def getText(datapath):
    f = open(datapath, 'r', encoding='utf-8')
    lists = f.readlines()
    # print(lists)
    text = []
    result = []
    for list in lists:
        if list != '\n':
            if list[-1] == '\n':
                list = list[0:-1]
            if list[-1]==' ':
                list = list[0:-1]
            text.append(list)

    return text
if __name__ =='__main__':
    text = getText('./data/TestHe.txt')
    print(text)
>>>>>>> 0788edf81b2bbb8ef6e66d48738a23171b113672
