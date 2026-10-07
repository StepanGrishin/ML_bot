import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


URL = "https://noproxy.tennisexplorer.com/next/"

BEST_OF_FIVE_KEYWORDS = (
    "australian open",
    "roland garros",
    "french open",
    "wimbledon",
    "us open",
)


class Match:
    def __init__(
        self,
        player_a,
        player_b,
        time="",
        tournament="",
        sets_a=None,
        sets_b=None,
        surface=None,
        date=None,
    ):
        self.player_a = player_a
        self.player_b = player_b
        self.time = time
        self.tournament = tournament
        self.sets_a = sets_a
        self.sets_b = sets_b
        self.surface = surface
        self.date = date

        self.finished = is_finished(
            tournament,
            sets_a,
            sets_b,
        )

    @property
    def winner(self):
        if not self.finished:
            return None

        if self.sets_a > self.sets_b:
            return self.player_a

        if self.sets_b > self.sets_a:
            return self.player_b

        return None

    def __repr__(self):
        return f"{self.player_a} - {self.player_b}"


def clean_player_name(name):
    name = re.sub(r"\(\d+\)", "", name or "")
    return " ".join(name.split()).strip()


def to_int(text):
    text = (text or "").strip()
    return int(text) if text.isdigit() else None


def is_finished(tournament, sets_a, sets_b):
    if sets_a is None or sets_b is None:
        return False

    need = 3 if any(
        k in (tournament or "").lower()
        for k in BEST_OF_FIVE_KEYWORDS
    ) else 2

    return max(sets_a, sets_b) >= need


def parse_matches(html, target_date):
    soup = BeautifulSoup(html, "html.parser")
    matches = []

    for box in soup.select("div.box.tbl"):
        tab = box.select_one("ul.tabs li.set span.tab")

        if not tab:
            continue

        if tab.get_text(strip=True) != target_date:
            continue

        tournament = ""
        pending = None

        for row in box.select("table.result tr"):
            classes = row.get("class") or []

            if "head" in classes:
                link = row.select_one("td.t-name a")

                tournament = (
                    link.get_text(strip=True)
                    if link
                    else ""
                )

                pending = None
                continue

            name_link = row.select_one("td.t-name a")

            if not name_link:
                continue

            name = clean_player_name(
                name_link.get_text(" ", strip=True)
            )

            result_cell = row.select_one("td.result")

            sets = to_int(
                result_cell.get_text(strip=True)
                if result_cell
                else ""
            )

            time_cell = row.select_one("td.time")

            if time_cell:
                pending = (
                    name,
                    time_cell.get_text(strip=True),
                    sets,
                )

            elif pending:
                matches.append(
                    Match(
                        player_a=pending[0],
                        player_b=name,
                        time=pending[1],
                        tournament=tournament,
                        sets_a=pending[2],
                        sets_b=sets,
                        date=target_date,
                    )
                )

                pending = None

    return matches


def get_matches_for_date(date):
    """
    Получает ATP-матчи на конкретную дату.
    """

    params = {
        "type": "atp-single",
        "year": date.year,
        "month": f"{date.month:02d}",
        "day": f"{date.day:02d}",
    }

    response = requests.get(
        URL,
        params=params,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64)"
            )
        },
    )

    response.raise_for_status()

    target_date = date.strftime("%d. %m. %Y")

    return parse_matches(
        response.text,
        target_date,
    )


def get_today_matches():
    """
    Совместимость со старым кодом.
    Возвращает матчи только на сегодня.
    """

    today = datetime.now().date()

    return get_matches_for_date(today)


def get_upcoming_matches(days=2):
    """
    Возвращает матчи на сегодня + следующие days дней.

    days=2:
        сегодня
        завтра
        послезавтра
    """

    today = datetime.now().date()

    all_matches = []

    for offset in range(days + 1):
        date = today + timedelta(days=offset)

        try:
            matches = get_matches_for_date(date)
            all_matches.extend(matches)

        except requests.RequestException as e:
            print(
                f"⚠️ Не удалось получить расписание "
                f"на {date}: {e}"
            )

    return all_matches
