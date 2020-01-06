from neural import train_and_check_model, predicting_model


def use_code(name_picture):
    type_weather = ['облачно', 'малооблачно', 'ясно', 'нет неба']
    prediction = predicting_model(name_picture)

    print(f'На этой фотографии {type_weather[prediction.argmax()]}')


use_code('sky.jpg')