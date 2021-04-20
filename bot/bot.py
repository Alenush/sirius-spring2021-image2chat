from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiofiles import os as aio_os

#import requests

from google_trans_new import google_translator

from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton


import aiohttp
import io
import logging
import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import typing as tp
from data_proc import data_processing
from config import TOKEN

import styles as st
import keyboards as kb
import translator as tr
TYPE = 'image'



bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

BOT_MES = ""

from google_trans_new import google_translator

bot_path = '//Users/isypov/Desktop/Bot/'

@dp.message_handler(commands=['1'])
async def process_command_1(message: types.Message):
    await message.reply("Друг, какой ты сегодня?", reply_markup=kb.inline_kb1)
    #await message.reply("Friend, what are you like today??", reply_markup=kb.inline_kb1)

@dp.callback_query_handler()
async def process_callback_kb1btn1(callback_query: types.CallbackQuery):
    code = callback_query.data[-1]
    if code.isdigit():
        code = int(code)
    if code == 1:
        print('Нажата злюка кнопка')
        #f = open(f'//Users/isypov/Desktop/Bot/{callback_query.from_user.id}/style.txt', 'w+')
        f = open(bot_path+f'{callback_query.from_user.id}/style.txt', 'w+')
        style_type = st.GetStyle(0)
        f.write(style_type)
        f.close()
        await bot.send_message(callback_query.from_user.id, "Ну ничего, со всеми бывает!\nДруг, отправь мне картинку, поговорим")
        #await bot.send_message(callback_query.from_user.id,
                               #"Well, nothing happens to everyone! Friend, send me a picture, let's talk")
    elif code == 2:
        print('Нажата середнячок кнопка')
        #f = open(f'//Users/isypov/Desktop/Bot/{callback_query.from_user.id}/style.txt', 'w+')
        f = open(bot_path + f'{callback_query.from_user.id}/style.txt', 'w+')
        style_type = st.GetStyle(1)
        f.write(style_type)
        f.close()
        await bot.send_message(callback_query.from_user.id,"Понимаю твой настрой!\nДруг, отправь мне картинку, поговорим")
    elif code == 3:
        #f = open(f'//Users/isypov/Desktop/Bot/{callback_query.from_user.id}/style.txt', 'w+')
        f = open(bot_path + f'{callback_query.from_user.id}/style.txt', 'w+')
        style_type = st.GetStyle(1)
        f.write(style_type)
        f.close()
        await bot.send_message(callback_query.from_user.id,
                               "Так держать!\nДруг, отправь мне картинку, поговорим")
        print('Нажата добряк кнопка')
    elif code == 4:
        #f4 = open(f'//Users/isypov/Desktop/Bot/{callback_query.from_user.id}/orig.txt', 'r')
        f4 = open(bot_path+f'{callback_query.from_user.id}/orig.txt', 'r')
        BOT_MES = f4.readline()
        f4.close()
        await bot.send_message(callback_query.from_user.id, BOT_MES)

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    st.GetStyle(2)
    #if not os.path.exists(f'//Users/isypov/Desktop/Bot/{message.from_user.id}'):
    #    os.makedirs(f'//Users/isypov/Desktop/Bot/{message.from_user.id}')
    #f = open(f'//Users/isypov/Desktop/Bot/{message.from_user.id}/dialog.txt', 'w+')
    if not os.path.exists(bot_path+f'{message.from_user.id}'):
        os.makedirs(bot_path+f'{message.from_user.id}')

    print(bot_path+f'{message.from_user.id}')
    f = open(bot_path+f'{message.from_user.id}/dialog.txt', 'w+')
    f.close()

    #f1 = open(f'//Users/isypov/Desktop/Bot/{message.from_user.id}/dialogeng.txt', 'w+')
    f1 = open(bot_path+f'{message.from_user.id}/dialogeng.txt', 'w+')
    f1.close()
    #await message.reply("Друг! Привет, я Зозо!\nКакой ты сегодня?", reply_markup=kb.inline_kb1)
    await message.reply("Debug", reply_markup=kb.inline_kb1)
    #await message.reply("Друг, какой ты сегодня?", reply_markup=kb.inline_kb1)


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply("Друг, отправь мне картинку и предложение, связанное с этой картинкой, и я отпрпавлю текст тебе в ответ!"
                        " Если хочешь начать всё с начала, набери /start")


@dp.message_handler()
async def echo_message(msg: types.Message):
    print(msg)
    dialog = msg.text
    #f = open(f'//Users/isypov/Desktop/Bot/{msg.from_user.id}/dialog.txt', 'a')
    f = open(bot_path+f'{msg.from_user.id}/dialog.txt', 'a')
    f.write(dialog+'\n')
    bot_mes = 'Why you trappin so hard?'

    #f1 = open(f'//Users/isypov/Desktop/Bot/{msg.from_user.id}/dialogeng.txt', 'a')
    f1 = open(bot_path+f'{msg.from_user.id}/dialogeng.txt', 'a')
    dialog = tr.translate_me(dialog)
    #tr.translate_text("привет друг", "en")
    # print("dialog: ", dialog)
    f1.write(dialog + '\n')
    f1.close()
    await bot.send_message(msg.from_user.id, "Дай подумать..")
    #bot_mes = data_processing(f'//Users/isypov/Desktop/Bot/{msg.from_user.id}/')
    bot_mes = data_processing(bot_path+f'{msg.from_user.id}/')
    #f4 = open(f'//Users/isypov/Desktop/Bot/{msg.from_user.id}/orig.txt', 'w+')
    f4 = open(bot_path+f'{msg.from_user.id}/orig.txt', 'w+')
    f4.write(bot_mes)
    f4.close()
    BOT_MES = bot_mes
    ru_bot_mes = tr.translate_me(bot_mes, 'ru')
    f.write(ru_bot_mes + '\n')
    f.close()
    #f1 = open(f'//Users/isypov/Desktop/Bot/{msg.from_user.id}/dialogeng.txt', 'a')
    f1 = open(bot_path+f'{msg.from_user.id}/dialogeng.txt', 'a')
    f1.write(bot_mes + '\n')
    f1.close()

    await bot.send_message(msg.from_user.id, ru_bot_mes)

    await msg.reply("Показать оригинальный ответ", reply_markup=kb.inline_kb2)

@dp.message_handler(content_types=['photo'])
async def handle_photo(message):
    #if not os.path.isdir(f'//Users/isypov/Desktop/Bot/{message.from_user.id}'):
    #    await aio_os.mkdir(f'//Users/isypov/Desktop/Bot/{message.from_user.id}')
    if not os.path.isdir(bot_path+f'{message.from_user.id}'):
        await aio_os.mkdir(bot_path+f'{message.from_user.id}')
    #await message.photo[-1].download(f'//Users/isypov/Desktop/Bot/{message.from_user.id}/{TYPE}.jpg')
    await message.photo[-1].download(bot_path+f'{message.from_user.id}/{TYPE}.jpg')
    if TYPE == 'image':
        await bot.send_message(message.chat.id, "Друг, вижу картинку! Что ты про неё думаешь?")


if __name__ == '__main__':
    executor.start_polling(dp)