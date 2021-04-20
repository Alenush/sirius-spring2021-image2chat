import random

bot_path = '//Users/isypov/Desktop/Bot/'

def GetStyle(num):
    if (num == 0):
        st = random.randint(0, 80)
        #st = 0;
        #f = open("/Users/isypov/Desktop/Bot/personalities.txt", "r")
        f = open(bot_path + "personalities.txt", "r")
        f.readline()
        for i in range(st):
            f.readline()
        style = f.readline()
        return (style)
    if (num == 1):
        st = random.randint(0, 97)
        #st = 0
        #f = open("/Users/isypov/Desktop/Bot/personalities.txt", "r")
        f = open(bot_path + "personalities.txt", "r")
        f.readline()
        for i in range(82+st):
            f.readline()
        style = f.readline()
        #print(style)
        return (style)
    if (num == 2):
        st = random.randint(0, 97)
        #st = 0
        #f = open("/Users/isypov/Desktop/Bot/personalities.txt", "r")
        f = open(bot_path + "personalities.txt", "r")
        f.readline()
        for i in range(119 + st):
            f.readline()
        style = f.readline()
        #print(style)
        return (style)



