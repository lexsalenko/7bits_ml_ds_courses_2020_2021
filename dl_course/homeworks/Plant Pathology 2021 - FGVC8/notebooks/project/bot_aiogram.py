import config
import logging
import model
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, executor, types

logging.basicConfig(level=logging.INFO)

bot = Bot(token=config.API_TOKEN)

dp = Dispatcher(bot)

photo_path = 'E:/Education/7bits/courses/MLandDS/ml-solutions/dl_course/homeworks/Plant Pathology 2021 - FGVC8/notebooks/photo/'

m = model.Model()

@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message):
    await message.photo[-1].download(photo_path + 'test.jpg')

    await message.answer('Фото получено.')

    await message.answer('Фото в обработке. Подождите несколько мгновений...')

    d = {'image': ['test.jpg'], 'labels': ['none']}

    df = pd.DataFrame(d)

    tmp = pd.DataFrame(np.zeros([len(df), 8]),
                       columns=['image', 'labels', 'scab', 'powdery_mildew', 'frog_eye_leaf_spot', 'complex', 'rust',
                                'healthy'])

    df = pd.concat([df, tmp], axis=1)

    result = m.get_prediction(df)

    await message.answer('Результат: ' + result)


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.answer("Привет!\nЯ Franz Liszt! Я детектирую болезни на листьях яблонь.\nДля работы отправь мне фото яблочного листа \nи я скажу тебе чем оно болеет.")


@dp.message_handler()
async def echo(message: types.Message):
    await message.answer(message.text)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
