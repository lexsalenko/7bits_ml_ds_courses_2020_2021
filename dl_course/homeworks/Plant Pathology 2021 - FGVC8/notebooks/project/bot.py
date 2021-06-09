import telebot
import config
import model
import random
import pandas as pd
import numpy as np


bot = telebot.TeleBot(config.TOKEN)


@bot.message_handler(commands=['start'])
def welcome(message):
    #sti = open('static/welcome.webp', 'rb')
    #bot.send_sticker(message.chat.id, sti)
 
    bot.send_message(message.chat.id, "Добро пожаловать, {0.first_name}!\nЯ - <b>{1.first_name}</b>, бот созданный чтобы быть подопытным кроликом.".format(message.from_user, bot.get_me()),
        parse_mode='html')



@bot.message_handler(content_types=['photo'])
def handle_docs_document(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = 'E:/Education/7bits/courses/MLandDS/ml-solutions/dl_course/homeworks/Plant Pathology 2021 - FGVC8/notebooks/photo/' + message.photo[1].file_id + '.jpg'
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    bot.reply_to(message, "Фото добавлено")

    d = {'image': ['href'], 'labels': ['none']}
    df = pd.DataFrame(d)

    tmp = pd.DataFrame(np.zeros([len(df), 8]), columns=['image', 'labels', 'scab', 'powdery_mildew', 'frog_eye_leaf_spot', 'complex', 'rust', 'healthy'])
    df = pd.concat([df, tmp], axis=1)

    m = model.Model()

    result = m.get_prediction(df)

    print(result)





# @bot.message_handler(content_types=['photo'])
# def image_handler(message):
#     #file = bot.getFile(update.message.photo[-1].file_id)
#     #update.message.reply_photo(photo=file)
    
#     chat_id = bot.get_updates()[-1].message.chat_id

#     pic = 'https://cutt.ly/dvjctaC'

#     bot.send_photo(chat_id, pic)


@bot.message_handler(content_types=['text'])
def lalala(message):
    bot.send_message(message.chat.id, message.text)


# RUN
bot.polling(none_stop=True)
