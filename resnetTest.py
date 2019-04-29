
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
from keras.models import load_model
import utils
import data_proprecession as dp
import class_dict


font = cv2.FONT_HERSHEY_SIMPLEX

model = ResNet50(
    weights=None,
    classes=25
)

if __name__ == '__main__':

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath,verbose=0, save_weights_only=True, mode="max", save_best_only=True)

    # model.fit(
    #     x=Images,
    #     y=labels,
    #     validation_split=0.1,
    #     epochs=50,
    #     batch_size=64,
    #     callbacks=[checkpoint]
    # )
    batch_size = 32
    model.fit_generator(dp.getpictures(batch_size),
                        samples_per_epoch=10000, nb_epoch=30, validation_data=0.1, max_q_size=1000,
                        verbose=1, nb_worker=1,callbacks=[checkpoint])

    model.save('my_model.h5')

    # imgpath = '/Users/peter/data/AI/test/72349.jpg'
    # root_path = '/Users/peter/data/AI/test'
    #
    # model = load_model('my_model.h5')
    #
    # code = utils.ImageEncode(imgpath)
    # ret = model.predict(code)
    # res1 = np.argmax(ret[0, :])
    # img = cv2.imread(imgpath)
    # classification = class_dict.class_dict[res1]
    # cv2.putText(img, str(res1), (50, 100), font, 2, (255, 255, 255), 7)
    # cv2.imshow('classification', img)
    #
    # cv2.waitKey(0)
    # image_name = imgpath.split('/')[-1].split('.')[0]
    # cv2.imwrite(root_path+'/'+image_name+'_predict.jpg')