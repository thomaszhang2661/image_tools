import os
import sys
import random

image_path = './images/'
label_path = os.path.join(image_path, 'labels')
if len(sys.argv) >= 2:
    label_path = sys.argv[1]
files = os.listdir(label_path)

ratio = 0.9
if len(sys.argv) >= 3:
    ratio = float(sys.argv[2])

invalid = []
listall = []
list1 = []
list2 = []

# textfile = open("invalid", "r")
# for element in textfile:
#     invalid.append(element.strip())

for file in files:
    if not file.endswith('.json'):
        continue

    imageid = file[:file.rfind(".")]
    ext = file[file.rfind("."):]

    if imageid not in invalid:
        listall.append(imageid)

count1 = len(listall) * ratio
random.shuffle(listall)

index = 0
for file in listall:
    if index >= count1:
        list2.append(file)
    else:
        list1.append(file)
    index = index + 1

print(len(listall))
print(len(list1))
print(list1)
print(len(list2))
print(list2)

textfile = open(os.path.join(image_path, "train.txt"), "w")
for element in list1:
    textfile.write(element + "\n")
textfile.close()

textfile = open(os.path.join(image_path, "val.txt"), "w")
for element in list2:
    textfile.write(element + "\n")
textfile.close()
