import asyncio
import logging
from datetime import datetime

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)

from config import BOT_TOKEN, MIN_COMMON_OPPONENTS
from predictor import TennisPredictor
from schedule import get_upcoming_matches


logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

predictor = TennisPredictor()

today_matches = {}


def infer_surface(tournament):
    name = (tournament or "").lower()

    if any(
        x in name
        for x in (
            "french open",
            "roland garros",
            "monte carlo",
            "rome",
            "madrid",
            "barcelona",
            "hamburg",
        )
    ):
        return "Clay"

    if "wimbledon" in name:
        return "Grass"

    return "Hard"


def analyze_today():
    LOOKAHEAD_DAYS = 2

    matches = get_upcoming_matches(days=LOOKAHEAD_DAYS)

    upcoming = []
    wins = 0
    losses = 0
    skipped = 0

    for match in matches:
        surface = getattr(
            match,
            "surface",
            None,
        ) or infer_surface(match.tournament)

        result = predictor.predict(
            match.player_a,
            match.player_b,
            surface=surface,
            before_date=today,
        )

        has_prediction = (
            result["common_count"]
            >= MIN_COMMON_OPPONENTS
        )

        if match.finished:
            predicted = result["winner"]
            actual = match.winner

            if has_prediction and predicted and actual:
                if predicted == actual:
                    wins += 1
                else:
                    losses += 1
            else:
                skipped += 1

        elif has_prediction and result["common_count"] > 0:
            upcoming.append((match, result))

    return upcoming, {
        "total": len(matches),
        "wins": wins,
        "losses": losses,
        "skipped": skipped,
    }


@dp.message(CommandStart())
async def start_handler(message: Message):
    await show_today_matches(message)


async def show_today_matches(message: Message):
    global today_matches

    try:
        upcoming, stats = await asyncio.to_thread(
            analyze_today
        )
    except Exception:
        logging.exception(
            "Ошибка анализа расписания"
        )
        await message.answer(
            "❌ Не удалось получить расписание "
            "или построить прогноз."
        )
        return

    if stats["total"] == 0:
        await message.answer(
            "Сегодня матчей не найдено."
        )
        return

    today_matches = {
        str(i): pair
        for i, pair in enumerate(upcoming)
    }

    keyboard = []

    for i, (match, _) in today_matches.items():
        keyboard.append([
            InlineKeyboardButton(
                text=(
                    f"🎾 {match.player_a} — "
                    f"{match.player_b}"
                ),
                callback_data=f"match:{i}",
            )
        ])

    keyboard.append([
        InlineKeyboardButton(
            text="🔄 Обновить",
            callback_data="refresh",
        )
    ])

    text = "🎾 <b>Матчи сегодня</b>\n\n"

    if upcoming:
        text += "Выбери матч для анализа."
    else:
        text += (
            "Предстоящих матчей с доступным "
            "прогнозом нет."
        )

    text += "\n\n" + format_stats(stats)

    await message.answer(
        text,
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=keyboard
        ),
        parse_mode="HTML",
    )


def format_stats(stats):
    wins = stats["wins"]
    losses = stats["losses"]
    skipped = stats["skipped"]
    done = wins + losses

    text = "📊 <b>Статистика за сегодня</b>\n"

    if done == 0:
        text += (
            "Пока нет завершённых матчей "
            "с прогнозом."
        )
    else:
        accuracy = wins / done * 100
        text += (
            f"✅ Угадали: <b>{wins}</b>\n"
            f"❌ Не угадали: <b>{losses}</b>\n"
            f"Точность: <b>{accuracy:.0f}%</b>"
        )

    if skipped:
        text += (
            f"\n<i>Без прогноза: {skipped}</i>"
        )

    return text


@dp.callback_query(F.data == "refresh")
async def refresh_handler(callback: CallbackQuery):
    await callback.answer("Обновляю...")
    await show_today_matches(
        callback.message
    )


@dp.callback_query(F.data.startswith("match:"))
async def match_handler(callback: CallbackQuery):
    await callback.answer()

    index = callback.data.split(":")[1]
    pair = today_matches.get(index)

    if not pair:
        await callback.message.answer(
            "❌ Матч устарел. Нажми /start"
        )
        return

    match, result = pair

    await callback.message.answer(
        format_prediction(match, result),
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[[
                InlineKeyboardButton(
                    text="⬅️ Назад",
                    callback_data="refresh",
                )
            ]]
        ),
        parse_mode="HTML",
    )


def format_prediction(match, result):
    a = result["player_a"]
    b = result["player_b"]

    pa = result["probability_a"]
    pb = result["probability_b"]

    winner = result["winner"]

    surface = (
        getattr(match, "surface", None)
        or infer_surface(match.tournament)
    )

    text = (
        f"🎾 <b>{a} — {b}</b>\n\n"
        f"🏆 {match.tournament or 'ATP'}\n"
        f"🎾 Покрытие: <b>{surface}</b>\n"
    )

    if match.time:
        text += f"🕐 {match.time}\n"

    text += (
        "\n🤖 <b>ML prediction</b>\n\n"
        f"🟢 {a}: <b>{pa:.1f}%</b>\n"
        f"🔵 {b}: <b>{pb:.1f}%</b>\n\n"
        f"📊 ELO: {result['elo_a']:.0f} — "
        f"{result['elo_b']:.0f}\n"
        f"🔥 Surface ELO: "
        f"{result['surface_elo_a']:.0f} — "
        f"{result['surface_elo_b']:.0f}\n"
        f"🤝 Общих соперников: "
        f"{result['common_count']}\n"
        f"⚔️ H2H score: "
        f"{result['h2h']:+.2f}\n\n"
        f"🏆 <b>Прогноз: {winner}</b>"
    )

    details = result.get("details") or []

    if details:
        text += "\n\n<b>Общие соперники:</b>"

        for item in details[:10]:
            text += (
                f"\n\n<b>{item['opponent']}</b>\n"
                f"{a}: {item['a_wins']}-"
                f"{item['a_losses']}\n"
                f"{b}: {item['b_wins']}-"
                f"{item['b_losses']}"
            )

    return text


async def main():
    print("Tennis ML predictor bot started...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
