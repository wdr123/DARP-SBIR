import pickle
with open("Train.pickle", "rb") as f:
    Image_Array_Train, Sketch_Array_Train, Image_Name_Train, Sketch_Name_Train = pickle.load(f)
with open("Test.pickle", "rb") as f:
    Image_Array_Test, Sketch_Array_Test, Image_Name_Test, Sketch_Name_Test = pickle.load(f)

with open('sketch_train.pickle',"rb") as f:
    Sketch_Name_Train1 = pickle.load(f)
with open('sketch_test.pickle',"rb") as f:
    Sketch_Name_Test1 = pickle.load(f)

count = 0
for i, sketch in enumerate(Sketch_Name_Train1):
    if sketch==Sketch_Name_Train[i]:
        count += 1

print(count)
print(len(Sketch_Name_Train))
