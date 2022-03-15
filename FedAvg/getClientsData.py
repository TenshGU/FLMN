import numpy as np

def file2matrix(filename):
    """
   Desc:
       导入训练数据
   parameters:
       filename: 数据文件路径
   return:
       数据矩阵 returnMat 和对应的类别 classLabelVector
   """
    fr = open(filename, mode='r')
    # 获得文件中的数据行的行数
    numberOfLines = len(fr.readlines())
    # 生成对应的空矩阵
    # 例如: zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0
    returnMat = np.zeros((numberOfLines, 6))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    classCount = {}
    for line in fr.readlines():
        # str.strip([chars]) --返回已移除字符串头尾指定字符所生成的新字符串
        line = line.strip()
        # 以 ' ' 切割字符串
        listFromLine = line.split(' ')
        # 每列的属性数据
        returnMat[index, :] = listFromLine[0:6]
        # 每列的类别数据，就是 label 标签数据
        label = int(listFromLine[-1])
        classCount[str(label)] = 1 if str(label) not in classCount else classCount[str(label)] + 1
        classLabelVector.append(label)
        index += 1

    weight = []
    for c in classCount.values():
        weight.append(c / numberOfLines)
    weight.sort(reverse=True)

    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector, weight
