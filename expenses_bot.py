import logging
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import re
from datetime import datetime
from aiogram.dispatcher.filters import Text
from aiogram.types import InputMediaPhoto
from datetime import datetime, timedelta, date
import calendar
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler(timezone="Asia/Dubai")

async def on_startup(dp):
    scheduler.start()

# За прошлый месяц: 1 число каждого месяца в 09:00
async def monthly_graph_job():
    start, end = last_month_range()
    await send_period_graph(bot, GROUP_ID, start, end)

scheduler.add_job(monthly_graph_job, trigger="cron", day=1, hour=10, minute=0)

# За прошлую неделю: каждый понедельник в 09:00
async def weekly_graph_job():
    start, end = last_week_range()
    await send_period_graph(bot, GROUP_ID, start, end)

scheduler.add_job(weekly_graph_job, trigger="cron", day_of_week="mon", hour=10, minute=0)

def last_month_range():
    today = datetime.today()
    first_day_this_month = today.replace(day=1)
    last_day_last_month = first_day_this_month - timedelta(days=1)
    first_day_last_month = last_day_last_month.replace(day=1)
    return first_day_last_month.date(), last_day_last_month.date()

def last_week_range():
    today = datetime.today()
    start = today - timedelta(days=today.weekday() + 7)  # прошлый понедельник
    end = start + timedelta(days=6)                        # прошлое воскресенье
    return start.date(), end.date()

async def send_period_graph(bot, group_id, start, end, category="ALL"):

    db = sqlite3.connect("expenses.db")
    df = pd.read_sql_query(
        "SELECT username, category, amount, created_at FROM expenses WHERE created_at BETWEEN ? AND ?",
        db,
        params=(start.isoformat(), (end + timedelta(days=1)).isoformat())  # включаем конец периода
    )

    if df.empty:
        await bot.send_message(group_id, f"Нет данных за период {start} — {end}")
        return

    # Файлы для альбома
    file1 = f"summary_{start}_{end}.png"
    file2 = f"daily_{start}_{end}.png"

    generate_summary_image(df, file1)
    generate_daily_line_image(df, start, end, file2)

    total_sum = df["amount"].sum()

    media = [
        InputMediaPhoto(
            media=types.InputFile(file1),
            caption=f"📊 Все категории\n📅 {start} — {end}\n💰 Общая сумма: {total_sum:.2f} ₽"
        ),
        InputMediaPhoto(media=types.InputFile(file2))
    ]

    await bot.send_media_group(chat_id=group_id, media=media)

    # Удаляем временные файлы
    os.remove(file1)
    os.remove(file2)


def prepare_daily_stats(df):
    df["date"] = pd.to_datetime(df["created_at"]).dt.date

    by_user_day = df.groupby(["date", "username"])["amount"].sum().unstack(fill_value=0)
    by_day_total = df.groupby("date")["amount"].sum()

    return by_user_day, by_day_total

def parse_dates(text: str):
    match = re.match(
        r"(\d{2}\.\d{2}\.\d{4})\s*-\s*(\d{2}\.\d{2}\.\d{4})",
        text
    )
    if not match:
        return None

    start = datetime.strptime(match.group(1), "%d.%m.%Y")
    end = datetime.strptime(match.group(2), "%d.%m.%Y")
    return start, end

def get_expenses_df(start, end):
    query = """
    SELECT username, category, amount, created_at
    FROM expenses
    WHERE created_at BETWEEN ? AND ?
    """
    df = pd.read_sql_query(
        query,
        db,
        params=(start.isoformat(), end.isoformat())
    )
    return df

def generate_summary_image(df, filename):
    by_user = df.groupby("username")["amount"].sum()
    by_category = df.groupby("category")["amount"].sum()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    by_user.plot(kind="bar")
    plt.title("Траты по пользователям")
    plt.ylabel("Сумма")

    plt.subplot(1, 2, 2)
    by_category.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Траты по категориям")
    plt.ylabel("")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_daily_line_image(df, start, end, filename):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    df["date"] = pd.to_datetime(df["created_at"])

    # диапазон корректно включает весь день
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())

    mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)
    df = df.loc[mask]

    if df.empty:
        plt.figure(figsize=(16, 6))
        plt.title("Нет данных за выбранный период")
        plt.savefig(filename)
        plt.close()
        return

    df["date_only"] = df["date"].dt.date
    all_days = pd.date_range(start=start_dt, end=end_dt).date

    users = df["username"].unique()
    by_user_day = df.pivot_table(
        index="date_only",
        columns="username",
        values="amount",
        aggfunc="sum",
        fill_value=0
    )

    by_user_day = by_user_day.reindex(all_days, fill_value=0)
    by_day_total = by_user_day.sum(axis=1)

    plt.figure(figsize=(16, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "D", "^", "v", "P"]

    for i, user in enumerate(users):
        plt.plot(
            by_user_day.index,
            by_user_day[user],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            label=user
        )

    plt.plot(
        by_day_total.index,
        by_day_total,
        marker=None,
        color="black",
        linewidth=2,
        linestyle="--",
        alpha=0.5,
        label="Общий"
    )

    plt.title("Траты по дням", fontsize=16)
    plt.xlabel("Дата", fontsize=12)
    plt.ylabel("Сумма", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename)
    plt.close()




async def build_and_send_graph(message, state, start, end):
    data = await state.get_data()
    category = data["category"]

    df = get_expenses_df(start, end)

    if category != "ALL":
        df = df[df["category"] == category]

    if df.empty:
        await message.answer("Нет данных за выбранный период")
        await state.finish()
        return

    total_sum = df["amount"].sum()

    file1 = f"summary_{message.from_user.id}.png"
    file2 = f"daily_{message.from_user.id}.png"

    generate_summary_image(df, file1)
    generate_daily_line_image(df, start, end, file2)

    # создаём коллекцию фото
    media = [
        InputMediaPhoto(
            media=types.InputFile(file1),
            caption=(
                f"📊 {category}\n"
                f"📅 {start.date()} — {end.date()}\n"
                f"💰 Общая сумма: {total_sum:.2f} ₽"
            )
        ),
        InputMediaPhoto(media=types.InputFile(file2))
    ]

    await message.answer_media_group(media)

    # удаляем временные файлы
    os.remove(file1)
    os.remove(file2)

    await state.finish()


def get_month_range(offset=0):
    today = datetime.today()
    first = (today.replace(day=1) - timedelta(days=offset*30)).replace(day=1)
    last = (first.replace(month=first.month % 12 + 1) - timedelta(days=1))
    return first, last

def graph_categories_kb():
    kb = types.InlineKeyboardMarkup(row_width=2)

    for c in categories:
        kb.insert(
            types.InlineKeyboardButton(
                text=c,
                callback_data=f"graph_cat:{c}"
            )
        )

    kb.add(
        types.InlineKeyboardButton(
            "📦 Все категории",
            callback_data="graph_cat:ALL"
        )
    )
    return kb

def graph_period_kb():
    return types.InlineKeyboardMarkup(
        inline_keyboard=[
            [types.InlineKeyboardButton("📅 Этот месяц", callback_data="graph_period:this_month")],
            [types.InlineKeyboardButton("📆 Прошлый месяц", callback_data="graph_period:last_month")],
            [types.InlineKeyboardButton("✍️ Ввести период вручную", callback_data="graph_period:manual")]
        ]
    )


API_TOKEN = os.getenv("API_TOKEN")
GROUP_ID = int(os.getenv("GROUP_ID"))
ALLOWED_USERS = set(map(int, os.getenv("ALLOWED_USERS").split(",")))

if not API_TOKEN:
    raise RuntimeError("API_TOKEN is not set")

if not GROUP_ID:
    raise RuntimeError("GROUP_ID is not set")

if not ALLOWED_USERS:
    raise RuntimeError("ALLOWED_USERS is not set")

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())

# -------------------- DATABASE --------------------
db = sqlite3.connect("expenses.db")
cur = db.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    username TEXT,
    category TEXT,
    amount REAL,
    created_at TEXT
)
""")
db.commit()

# -------------------- FSM --------------------
class GraphFSM(StatesGroup):
    choose_category = State()
    choose_period = State()
    manual_dates = State()

class ExpenseState(StatesGroup):
    waiting_amount = State()
    confirm = State()

# -------------------- KEYBOARDS --------------------
main_kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
categories = [
    "Вкусняшки", "Продукты", "Доставка",
    "Развлечения", "Косметика",
    "Быт", "Курительные"
]
for c in categories:
    main_kb.add(c)
main_kb.add("📊 Графики")

cancel_kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
cancel_kb.add("/cancel")

confirm_kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
confirm_kb.add("✅ Да", "❌ Нет")

# -------------------- HELPERS --------------------
def check_access(message: types.Message):
    return message.from_user.id in ALLOWED_USERS

# -------------------- HANDLERS --------------------
@dp.message_handler(commands="start")
async def start(message: types.Message):
    if not check_access(message):
        return

    await message.answer("Выбери категорию:", reply_markup=main_kb)

@dp.message_handler(commands="cancel", state="*")
async def cancel(message: types.Message, state: FSMContext):
    await state.finish()
    await message.answer("Действие отменено", reply_markup=main_kb)

@dp.message_handler(Text(equals="📊 Графики"))
async def graph_start(message: types.Message, state: FSMContext):
    await GraphFSM.choose_category.set()
    await message.answer(
        "Выбери категорию:",
        reply_markup=graph_categories_kb()
    )

@dp.callback_query_handler(lambda c: c.data.startswith("graph_cat"), state=GraphFSM.choose_category)
async def graph_choose_category(call: types.CallbackQuery, state: FSMContext):
    category = call.data.split(":")[1]

    await state.update_data(category=category)
    await GraphFSM.choose_period.set()

    await call.message.edit_text(
        "За какой период построить график?",
        reply_markup=graph_period_kb()
    )

@dp.callback_query_handler(lambda c: c.data.startswith("graph_period"), state=GraphFSM.choose_period)
async def graph_choose_period(call: types.CallbackQuery, state: FSMContext):
    period = call.data.split(":")[1]

    if period == "manual":
        await GraphFSM.manual_dates.set()
        await call.message.answer(
            "Введи период в формате:\n"
            "20.11.2024 - 11.12.2025"
        )
        return

    if period == "this_month":
        start, end = get_month_range(0)

    elif period == "last_month":
        start, end = get_month_range(1)

    await build_and_send_graph(call.message, state, start, end)


@dp.message_handler(state=GraphFSM.manual_dates)
async def graph_manual_dates(message: types.Message, state: FSMContext):
    dates = parse_dates(message.text)
    if not dates:
        await message.answer("Неверный формат дат")
        return

    start, end = dates
    await build_and_send_graph(message, state, start, end)

@dp.message_handler(lambda m: m.text in categories)
async def choose_category(message: types.Message, state: FSMContext):
    if not check_access(message):
        return

    await state.update_data(category=message.text)
    await ExpenseState.waiting_amount.set()
    await message.answer(
        f"Введи сумму для категории «{message.text}»",
        reply_markup=cancel_kb
    )

@dp.message_handler(state=ExpenseState.waiting_amount)
async def enter_amount(message: types.Message, state: FSMContext):
    try:
        amount = float(message.text.replace(",", "."))
    except:
        await message.answer("Введи сумму")
        return

    await state.update_data(amount=amount)
    await ExpenseState.confirm.set()

    await message.answer(
        f"Сумма {amount} верна?",
        reply_markup=confirm_kb
    )

@dp.message_handler(lambda m: m.text == "❌ Нет", state=ExpenseState.confirm)
async def decline(message: types.Message, state: FSMContext):
    await state.finish()
    await message.answer("Окей, начнём заново", reply_markup=main_kb)

@dp.message_handler(lambda m: m.text == "✅ Да", state=ExpenseState.confirm)
async def save(message: types.Message, state: FSMContext):
    data = await state.get_data()

    cur.execute(
        "INSERT INTO expenses VALUES (NULL, ?, ?, ?, ?, ?)",
        (
            message.from_user.id,
            message.from_user.username,
            data["category"],
            data["amount"],
            datetime.now().isoformat()
        )
    )
    db.commit()

    await bot.send_message(
        GROUP_ID,
        f"💸 {data['category']}: {data['amount']} ₽"
    )

    await state.finish()
    await message.answer("Сохранено ✅", reply_markup=main_kb)

# -------------------- START --------------------
if __name__ == "__main__":
    executor.start_polling(
        dp,
        skip_updates=True,
        on_startup=on_startup
    )
