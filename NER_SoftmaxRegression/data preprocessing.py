# coding=gbk
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import collections

# 作为特征表示的向量维度
common_word_length = 1000

# 词及其标签的读入
def read_vector(single_vector, total):
    last_temp = 1e9
    for t in range(len(single_vector)):  # 词列表
        if single_vector[t] == '{':  # 去除注音
            last_temp = t
            continue
        if single_vector[t] == '/':
            vector_last = min(t, last_temp)
            if single_vector[0] != '[':
                vec.append(single_vector[0:vector_last])
            else:
                vec.append(single_vector[1:vector_last])
                square_start.append(total)  # 对方框特殊处理
            flag = True

            for j in range(t + 1, len(single_vector)):  # 标签列表
                if single_vector[j] == ']':
                    label.append(single_vector[t + 1:j])
                    while len(square_start) > len(square_end) + 1:  # 消除数据噪声
                        square_start.pop()
                    while len(square_end) > len(square_start) - 1 and len(square_end) != 0:
                        square_end.pop()
                        square_label.pop()
                    square_label.append(single_vector[j + 1:])
                    square_end.append(total)
                    flag = False
                    break
            if flag:
                label.append(single_vector[t + 1:])
            break


# 词频统计
def count_frequency(Dict, str):
    if str in Dict:
        Dict[str] += 1
    else:
        Dict[str] = 1


# 输出常用1000词：三种类型的BI+other各250个
def count_top500():
    # 加载停用词
    with open("data/stop_words", encoding="utf-8-sig") as f:
        stop_words = f.read().split()
    stop_words.extend(['〓','５', '１', '４', '３', '２', '１００', '６', '１０', '２０',  '１９９７年'])
    for word in stop_words:
        if word in words_other:
            del words_other[word]  # 删除对应的词
        if word in words_name:
            del words_name[word]
        if word in words_place:
            del words_place[word]
        if word in words_organization:
            del words_organization[word]
    aver_num = int(common_word_length/4)
    wordsDic_seq = collections.Counter(words_name).most_common(aver_num) + collections.Counter(words_place).most_common(
        aver_num) + collections.Counter(words_organization).most_common(aver_num) + collections.Counter(words_other).most_common(
        aver_num-1)
    '''
    #全部词中选取出现频率最高的词
    words = {}
    # words = dict(words_name.items() +words_place.items()+words_organization.items() +words_other.items())
    words.update(words_name)
    words.update(words_place)
    words.update(words_organization)
    words.update(words_other)
    wordsDic_seq = collections.Counter(words).most_common(800)
    '''
    for x, y in wordsDic_seq:
        print("'" + x + "',", file=f4, end='')
    return wordsDic_seq


# 生成y标签
def generate_y_label():  # 人名0,1；地名2,3；机构4,5,；其他6
    count = 0
    current = 0
    while count < len(vec):
        if current < len(square_start) - 1 and count == square_start[current]:
            if square_label[current] == 'ns':
                one_hot.append('2')
                count_frequency(words_place, vec[count])
                count += 1
                while count <= square_end[current]:
                    one_hot.append('3')
                    count_frequency(words_place, vec[count])
                    count += 1
            elif square_label[current] == 'nt':
                one_hot.append('4')
                count_frequency(words_organization, vec[count])
                count += 1
                while count <= square_end[current]:
                    one_hot.append('5')
                    count_frequency(words_organization, vec[count])
                    count += 1  
            if current < len(square_start) - 1:
                current += 1
                continue 
        # 生成标签
        if (label[count] == 'nr') or (label[count] == 'nrf' and count != 0 and label[count - 1] != 'nrf') or (
                label[count] == 'nrf' and count == 0):
            one_hot.append('0')
            count_frequency(words_name, vec[count])
        elif (label[count] == 'nrg') or (label[count] == 'nrf') or \
                (count != 0 and (label[count] == 'n' or label[count] == 'Ng') and (
                        label[count - 1] == 'nrf' or label[count - 1] == 'nrg')):
            one_hot.append('1')
            count_frequency(words_name, vec[count])
        elif label[count] == 'ns' and label[count - 1] != 'ns':
            one_hot.append('2')
            count_frequency(words_place, vec[count])
        elif label[count] == 'ns':
            one_hot.append('3')
            count_frequency(words_place, vec[count])
        elif label[count] == 'm' and label[count + 1] == 'q' and (
                one_hot[count - 1] == '3' or one_hot[count - 1] == '2'):
            one_hot.append('3')
            one_hot.append('3')
            count_frequency(words_place, vec[count])
            count_frequency(words_place, vec[count + 1])
            count += 2
            continue
        elif label[count] == 'nt':
            one_hot.append('4')
            count_frequency(words_organization, vec[count])
        else:
            one_hot.append('6')
            count_frequency(words_other, vec[count])
        count += 1
    return count


# 单个词的one_hot表示，先搜字典，字典里没有，再创建one-hot，前后拼接起来后再写入文件
def One_hot(str):
    if str in words500:
        return words500[str]
    else:
        global name_500
        one_hot_list = [0 for col in range(common_word_length)]
        flag = True
        for i in range(common_word_length - 1):
            if name_500[i] == str:
                flag = False
                one_hot_list[i] = 1
                break
        if flag:
            one_hot_list[common_word_length - 1] = 1
        words500[str] = one_hot_list
        return one_hot_list


# 用于训练的输入生成
def generate_x_one_hot(flag_t,one_hot_x,label_y):
    one_hot_zero = [0] * common_word_length
    cnt = 0
    for j in range(len(vec)):
        if j == 0:
            one_hot_x.append(one_hot_zero + One_hot(vec[j]) + One_hot(vec[j + 1]))
            label_y.append(one_hot[j])
        elif j == len(vec) - 1:
            one_hot_x.append(One_hot(vec[j - 1]) + One_hot(vec[j]) + one_hot_zero)
            label_y.append(one_hot[j])
        else:
            if flag_t == 1:
                temp1 = One_hot(vec[j])
                if one_hot[j] != '6' or temp1[common_word_length - 1] != 1  or one_hot[j-1] != '6' or one_hot[j+1] != '6': 
                    one_hot_x.append(One_hot(vec[j - 1]) + One_hot(vec[j]) + One_hot(vec[j + 1]))
                    label_y.append(one_hot[j])
                elif cnt <= 30000: # 保留O的个数
                    one_hot_x.append(One_hot(vec[j - 1]) + One_hot(vec[j]) + One_hot(vec[j + 1]))
                    label_y.append(one_hot[j])
                    cnt += 1
            else:
                one_hot_x.append(One_hot(vec[j - 1]) + One_hot(vec[j]) + One_hot(vec[j + 1]))
                label_y.append(one_hot[j])


# 制作词云
def word_cloud(wordsDic_seq):
    fname_mask = 'data/map.jpeg'
    fname_font = 'C:/Windows/Fonts/simkai.ttf'
    # processing image
    im_mask = np.array(Image.open(fname_mask))
    im_colors = ImageColorGenerator(im_mask)
    # generate word cloud
    wcd = WordCloud(font_path=fname_font,  # font for Chinese charactors
                    background_color='white',
                    mode="RGBA",
                    max_words=1500,
                    mask=im_mask,
                    )
    wordsDic = {}
    for x, y in wordsDic_seq:
        wordsDic[x] = y
    wcd.generate_from_frequencies(wordsDic)
    wcd.recolor(color_func=im_colors)

    def plt_imshow(x, ax=None, show=True):
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(x)
        ax.axis("off")
        if show: plt.show()
        # ax.figure.savefig(f'conbined_wcd.png', bbox_inches='tight', dpi=150)
        return ax

    ax = plt_imshow(wcd, )
    ax.figure.savefig(f'result/wordcloud.png', bbox_inches='tight', dpi=800)


def output_train():
    # 写文件
    x_train = np.array(one_hot_x_train)
    t_train = np.array(label_y_train)
    filename_x = 'data/x_train.npy'
    filename_y = 'data/t_train.npy'
    np.save(filename_x, x_train)
    np.save(filename_y, t_train)
    print("训练集已生成完毕!","输入X大小：",x_train.shape, "标签t大小：",t_train.shape)


def output_test():
    x_test = np.array(one_hot_x_dev)
    t_test = np.array(label_y_dev)
    filename_x = 'data/x_test.npy'
    filename_y = 'data/t_test.npy'
    np.save(filename_x, x_test)
    np.save(filename_y, t_test)
    print("验证集已生成完毕!","输入X大小：",x_test.shape,"标签t大小：", t_test.shape)


def output_final():
    x_test = np.array(one_hot_x_test)
    t_test = np.array(label_y_test)
    filename_x = 'data/x_final.npy'
    filename_y = 'data/t_final.npy'
    np.save(filename_x, x_test)
    np.save(filename_y, t_test)
    print("测试集已生成完毕!","输入X大小：",x_test.shape, "标签t大小：",t_test.shape)
    print("程序已执行完毕，可以停止运行")
    exit(1)


def initialize():
    global vec, label, one_hot, square_start, square_end, square_label, total
    vec = list()  # 词
    label = list()  # 对应标签，如nrg
    one_hot = list()  # 词的y标签
    square_start = list()  # 对于含方框的词需要特殊标记，为开始位置
    square_end = list()  # 标记结束位置
    square_label = list()  # 含方框的标签
    total = 0  # 统计词的个数


# 文件读写
f1 = open("data/f1.txt", 'r+', encoding='GBK')
f4 = open("result/name_top500.txt", 'w', encoding='GBK')

# 初始化
vec = list()  # 词
label = list()  # 对应标签，如nrg
one_hot = list()  # 词的y标签
square_start = list()  # 对于含方框的词需要特殊标记，为开始位置
square_end = list()  # 标记结束位置
square_label = list()  # 含方框的标签
total = 0  # 统计词的个数
words_name = {}  # 新建字典，统计词频
words_place = {}  # 新建字典，统计词频
words_organization = {}
words_other = {}
wordsDic_seq = {}  # 常用词
words500 = {}  # 每个词的500维one-hot表示

train_flag = True
dev_flag = True

for i in f1.readlines():
    if i[:8] == '19980121' and train_flag == True: # 训练集生成
        generate_y_label() # 生成y标签
        wordsDic_seq = count_top500() # 提取训练词中的特征（800词词典）
        name_500 = list()
        for x, y in wordsDic_seq: # 将词典转为列表，方便向量生成
            name_500.append(x)
        word_cloud(wordsDic_seq) # 生成词云
        train_flag = False
        one_hot_x_train = list()  
        label_y_train = list()  
        generate_x_one_hot(1,one_hot_x_train,label_y_train) # 生成词所对应的向量与标签对应的数字
        output_train() #写入文件
        initialize() #初始化，方便后续验证集、测试集生成
    elif i[:8] == '19980126' and dev_flag == True: # 验证集生成
        generate_y_label()
        dev_flag = False
        one_hot_x_dev = list()  
        label_y_dev = list()  
        generate_x_one_hot(0,one_hot_x_dev,label_y_dev)
        output_test()
        initialize()
    if i[0] == '\n':
        continue
    line = i.split()
    #读入每个词及其所对应的标签
    for count in range(len(line)):
        single_vector = line[count]
        read_vector(single_vector, total)
        total += 1

#测试集生成
generate_y_label()
one_hot_x_test = list()  # x
label_y_test = list()  # y
generate_x_one_hot(0,one_hot_x_test,label_y_test)
output_final()

f1.close()
f4.close()

