from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

button1 = KeyboardButton('Грустный')
button2 = KeyboardButton('Середнячок')
button3 = KeyboardButton('Веселый')

markup3 = ReplyKeyboardMarkup(
    resize_keyboard=True,
    one_time_keyboard=True
).add(button1).add(button2).add(button3)

inline_btn_1 = InlineKeyboardButton('Злой(', callback_data='btn1')
inline_btn_2 = InlineKeyboardButton('Ну так, средне', callback_data='btn2')
inline_btn_3 = InlineKeyboardButton('Добрый', callback_data='btn3')
inline_kb1 = InlineKeyboardMarkup().add(inline_btn_1).add(inline_btn_2).add(inline_btn_3)

inline_btn_4 = InlineKeyboardButton('Показать оригинальный ответ', callback_data='btn4')
inline_kb2 = InlineKeyboardMarkup().add(inline_btn_4)
