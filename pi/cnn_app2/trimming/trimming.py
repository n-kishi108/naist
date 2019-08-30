# http://qiita.com/Shiratah/items/a1dab6c6b5ba088123e0
import cv2

# http://qiita.com/suppy193/items/91609e75789e9f458c39
#cascade_path = "./face.xml"
# http://ultraist.hatenablog.com/entry/20110718/1310965532
cascade_path = "./lbpcascade_animeface.xml"
origin_image_path = "./orig/"
dir_path = "./dist/"

i = 0

for line in open('./filename.txt','r'):
    line = line.rstrip()
    print(line)
    #image = cv2.imread(origin_image_path+line,0)
    image = cv2.imread(origin_image_path+line)
    if image is None:
        print('Not open : ',line)
        quit()

    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))

    if len(facerect) > 0:
        for rect in facerect:
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            dst = image[y:y + height, x:x + width]
            #save_path = dir_path + '/' + 'image(' + str(i) + ')' + '.jpg'
            save_path = dir_path + '/' + str(i) + '_' + line
            cv2.imwrite(save_path, dst)
            print("save!")
            i += 1
print("Finish")
