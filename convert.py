#conding=utf8  
import os
from PIL import Image

g = os.walk(r"result")

for path,dir_list,file_list in g:  
    for file_name in file_list:
        # print(os.path.join(path, file_name))
        if file_name[-3:] == "eps":
            print(os.path.join(path, file_name) )
            new_file_name = os.path.join(path, file_name)
            a = Image.open(new_file_name).save(new_file_name[:-3]+"png")
