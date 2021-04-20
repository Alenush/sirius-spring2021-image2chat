import os.path
from bot_interaction import AskModel

def data_processing(data_path):
    image_path = data_path + "image.jpg"
    print("im:", image_path)
    if (os.path.exists(image_path) == False):
        return "no image";
    dial_path = data_path + "dialogeng.txt"
    #print("dial:", dial_path)
    style_path = data_path + "style.txt"
    #print(style_path)


    f3 = open(style_path, 'r')
    style_type = f3.readline()
    f3.close()
    #print(style_type)

    dial_story = ""
    f3 = open(dial_path, 'r')
    for line in f3:
        dial_story += line
    f3.close()
    print("dial story:", dial_story)
    ANSWER = AskModel(image_path, style_type, dial_story)

    #print("dial_story:", dial_story)
    return ANSWER