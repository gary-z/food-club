from bs4 import BeautifulSoup
from pirates import courses, pirates, arenas

days = []

for i in range(3600, 4900 + 1):
    with open("historical/Match_3600.txt", "r", encoding="utf-8") as f:
        content = f.read()
        soup = BeautifulSoup(markup=content, features="html.parser")
    table = soup.find(name="table")
    menus = []
    for menu_soup in soup.find_all(attrs={"class": "foods"}):
        foods = menu_soup.get_text().split(", ")
        assert len(foods) == 10
        menus.append(foods)
    assert len(menus) == 5

    pirate_names = []
    odds = []

    for row in table.find_all(name="tr"):
        datas = list(row.find_all(name="td"))
        if len(datas) < 9:
            continue
        if len(datas) > 9:
            datas = datas[1:-1]
        pirate_name = datas[0].contents[0].get_text()
        pirate_names.append(pirate_name)
        assert any(pirate.name == pirate_name for pirate in pirates)
        odds_text = datas[-4].contents[0].get_text()
        a, b = odds_text.split(":")
        a = int(a)
        odds.append(a)
    assert len(odds) == 20

    winners = []
    for winner_soup in soup.find_all(attrs={"class": "winner"}):
        winner = winner_soup.get_text()
        assert winner in pirate_names
        winners.append(winner)
    assert len(winners) == 5

    arena_datas = []
    for i, arena in enumerate(arenas):
        pirates_in_arena = pirate_names[i * 4 : i * 4 + 4]
        odds_in_arena = odds[i * 4 : i * 4 + 4]
        winner = winners[i]
        foods = menus[i]

        pirates_data = [
            {"name": pirates_in_arena[j], "odds": odds_in_arena[j]} for j in range(4)
        ]
        arena_datas.append(
            {
                "pirates": pirates_data,
                "winner": winner,
                "foods": foods,
                "arena_name": arena,
            }
        )
    days.append(arena_datas)
    # assert len(menus) == 5
    # print(foods)
    # arenas = list(table.children)[1:]
    # assert len(arenas) == 5
    # break

import json

with open("output.txt", "w", encoding="utf-8") as f:
    json.dump(days, f, indent=4)
