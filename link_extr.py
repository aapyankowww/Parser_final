import csv
import json
import logging
import random
import threading
import time
from pathlib import Path
import keyboard
from playwright.sync_api import sync_playwright, TimeoutError
from playwright_stealth import stealth_sync
from tabulate import tabulate
from tqdm import tqdm

# Global defaults
BTM_TEXT = 'Скачайте приложение'
CSV_FILE = 'links.csv'
MODELS_FILE = 'car_models.csv'
TRUCK_FILE = 'truck_models.csv'
COOKIES_FILE = 'cookies.json'
STORAGE_FILE = 'storage_state.json'
MAX_PAGES = 20
BATCH_SIZE = 50
VIEWPORT = {'width': 1366, 'height': 768}
AGENTS_FILE = 'user_agents.csv'
Path("logs").mkdir(parents=True, exist_ok=True)
formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
fh = logging.FileHandler('logs/extractor.log', encoding='utf-8', mode='a')
fh.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(ch)
logger.addHandler(fh)

stop_event = threading.Event()
keyboard.on_press_key('f12', lambda _: stop_event.set())

def log_section(title: str):
    logger.info('=== %s ===', title)

def format_models_table(models: list[dict]) -> str:
    if not models:
        return "— нет записей —"
    headers = models[0].keys()
    rows = [list(m.values()) for m in models]
    return tabulate(rows, headers=headers, tablefmt="psql", stralign="center")


# для безопасной загрузки CSV
def safe_load_csv(file_path: str) -> list[dict]:
    path = Path(file_path)
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows


# Links module
def load_existing_links() -> set[str]:
    if not Path(CSV_FILE).exists():
        return set()
    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if rows and rows[0] == ['link', 'status']:
        rows = rows[1:]
    return {row[0] for row in rows if row}

def append_links(links: list[str]) -> None:
    file_exists = Path(CSV_FILE).exists()
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['link', 'status'])
        writer.writerows((link, '') for link in links)


# Model file module
def load_models() -> list[dict]:
    return safe_load_csv(MODELS_FILE)

def load_models_truck() -> list[dict]:
    return safe_load_csv(TRUCK_FILE)

def update_models_file(processed: dict[str, int]) -> None:
    temp = Path(MODELS_FILE).with_suffix('.tmp')
    rows = load_models()
    fieldnames = rows[0].keys() if rows else ['brand', 'model', 'status']
    with open(temp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            key = f"{row['brand']} {row['model']}"
            # Считываем существующий статус, по умолчанию 0
            old_status = int(row.get('status', '0') or 0)
            # Если есть новые ссылки — суммируем
            if key in processed:
                new_status = old_status + processed[key]
            else:
                new_status = old_status
            row['status'] = str(new_status)
            writer.writerow(row)
    temp.replace(MODELS_FILE)


def update_truck_models_file(processed: dict[str, int]) -> None:
    temp = Path(TRUCK_FILE).with_suffix('.tmp')
    rows = load_models_truck()
    fieldnames = rows[0].keys() if rows else ['brand', 'model', 'status']
    with open(temp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            key = f"{row['brand']} {row['model']}"
            # Считываем существующий статус, по умолчанию 0
            old_status = int(row.get('status', '0') or 0)
            # Если есть новые ссылки — суммируем
            if key in processed:
                new_status = old_status + processed[key]
            else:
                new_status = old_status
            row['status'] = str(new_status)
            writer.writerow(row)
    temp.replace(TRUCK_FILE)


# Cookies
def load_cookies(ctx):
    if Path(STORAGE_FILE).exists():
        ctx.add_init_script(f"""() => {{
            const ls = {json.dumps(Path(STORAGE_FILE).read_text(encoding='utf-8'))};
        }}""")
    if Path(COOKIES_FILE).exists():
        with open(COOKIES_FILE, encoding='utf-8') as f:
            ctx.add_cookies(json.load(f))

def save_cookies(ctx):
    with open(COOKIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(ctx.cookies(), f, indent=2)
    try:
        ctx.storage_state(path=STORAGE_FILE)
    except Exception as e:
        logger.warning("Не удалось сохранить storage_state: %s", e)


# Links extractor helpers
def detect_site(url: str) -> str:
    if 'avito.ru' in url:
        return 'avito'
    if 'auto.ru' in url:
        return 'auto'
    raise ValueError('URL должен быть от avito.ru или auto.ru')

def get_selector(site: str) -> str:
    return (
        "//a[@data-marker='item-title']"
        if site == 'avito'
        else "//a[contains(@class,'ListingItemTitle__link')]"
    )

def get_pagination_param(site: str) -> str:
    return '&p=' if site == 'avito' else '?page='

def extract_links_from_elements(elems, site, collected, known, limit):
    new = []
    allowed = ['avtomobili', 'gruzoviki_i_spetstehnika']
    for el in elems:
        href = el.get_attribute('href') or ''
        if site == 'avito':
            full = 'https://www.avito.ru' + href.split('?')[0]
            if not any(sub in full for sub in allowed):
                continue
        else:
            full = href.split('?')[0]
        if full in collected or full in known:
            continue
        collected.add(full)
        new.append(full)
        if len(collected) >= limit:
            break
    return new

def mouse_pattern_z(page):
    w, h = VIEWPORT['width'], VIEWPORT['height']
    dx = 0.2 * w
    dy = 0.35 * h
    x, y = 0.45 * w, 0.05 * h
    for i in range(6):
        page.mouse.move(x, y, steps=20)
        time.sleep(random.uniform(1.5, 2.5))
        x -= dx * (-1) ** i
        y += dy * (1 - (-1) ** i) / 2

def scroll_and_collect(page, site, collected, known, pbar, limit):
    batch = []
    bottom = page.get_by_text(BTM_TEXT)
    box = bottom.bounding_box()
    for _ in range(10):
        if stop_event.is_set():
            return batch, True
        page.mouse.wheel(0, box['y']*0.08)
        time.sleep(random.uniform(1.2, 2.0))
        mouse_pattern_z(page)
        elements = page.locator(get_selector(site)).all()
        new = extract_links_from_elements(elements, site, collected, known, limit)
        pbar.update(len(new))
        batch.extend(new)
        if len(collected) >= limit:
            break
    return batch, False

def page_not_found(page, site) -> bool:
    if site == 'avito':
        return (
            page.locator('h1', has_text='Такой страницы нe существует').count() > 0
            or page.locator('h2', has_text='Ничего не найдено').count() > 0
        )
    return (
        page.locator('div.page-title', has_text='Страница не найдена').count() > 0
    )

# генераторы URL для авто и грузовиков
def generate_search_urls(site: str) -> list[tuple[str, str]]:
    base = {
        'avito': 'https://www.avito.ru/all/avtomobili/',
        'auto': 'https://auto.ru/rossiya/cars/'
    }[site]
    urls: list[tuple[str, str]] = []
    for row in load_models():
        #if int(row.get('status', '0')) < N_LINKS:
        q = f"{row['brand'].lower()}/{row['model'].lower()}".replace(' ', '_')
        url = base + q + ('/all' if site == 'auto' else '')
        urls.append((url, f"{row['brand']} {row['model']}"))
    return urls

def generate_search_urls_cars(site: str) -> list[tuple[str, str]]:
    if site == 'auto':
        return []
    base = 'https://www.avito.ru/all/avtomobili?cd=1'
    urls: list[tuple[str, str]] = []
    for row in load_models():
        #if int(row.get('status', '0')) < N_LINKS:
        q = f"&q={row['brand'].lower()}+{row['model'].lower()}"
        urls.append((base + q, f"{row['brand']} {row['model']}"))
    return urls

def generate_search_urls_truck(site: str) -> list[tuple[str, str]]:
    if site == 'auto':
        return []
    base = 'https://www.avito.ru/all/gruzoviki_i_spetstehnika?cd=1'
    urls: list[tuple[str, str]] = []
    for row in load_models_truck():
        #if int(row.get('status', '0')) < N_LINKS:
        q = f"&q={row['brand'].lower()}+{row['model'].lower()}"
        urls.append((base + q, f"{row['brand']} {row['model']}"))
    return urls

def check_model(name, batch):
    checked = []
    for link in batch:
        if name.lower().replace(' ', '_') in link:
            checked.append(link)
    return checked

def process_target(page, base_url, name, known, limit, processed):
    site = detect_site(base_url)
    param = get_pagination_param(site)
    collected = set()
    buffer = []

    log_section(f"START {name or base_url}")
    start_time = time.time()

    with tqdm(total=limit, desc='Сбор ссылок', ncols=80) as pbar:
        for i in range(1, MAX_PAGES + 1):
            if stop_event.is_set() or len(collected) >= limit:
                break
            url = base_url if i == 1 else f"{base_url}{param}{i}"
            try:
                resp = page.goto(url, timeout=60000)
            except TimeoutError:
                tqdm.write(f"\nТаймаут {url}")
                continue
            time.sleep(1)
            if page_not_found(page, site) or (resp and resp.status >= 400):
                break

            if page.locator(get_selector(site)).count() == 0:
                break

            batch, esc = scroll_and_collect(
                page, site, collected, known, pbar, limit
            )
            batch = check_model(name, batch)
            buffer.extend(batch)
            if esc:
                break
            if len(buffer) >= BATCH_SIZE:
                append_links(buffer)
                buffer.clear()
            time.sleep(random.uniform(3.5, 7.0))

    if buffer:
        append_links(buffer)
    if name:
        processed[name] = len(collected)
    known.update(collected)

    elapsed = time.time() - start_time
    logger.info(
        "Processed: %s → links: %d, time: %.2f s",
        name or base_url, len(collected), elapsed
    )
    log_section(f"END {name or base_url}")


def main(mode='csv', n_links = 10, max_links=100, site='avito'):
    log_section("AGENT START")
    overall_start = time.time()
    processed_counts = {}
    known_links = load_existing_links()

    with open(AGENTS_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        agents = [row[0] for row in reader]
    USER_AGENT = random.choice(agents)
    # Режим 4: логин и сохранение куки
    if mode == 'debug':
        with sync_playwright() as p:
            browser = p.chromium.launch(
                channel='chrome', headless=False, slow_mo=40
            )
            ctx = browser.new_context(
                viewport=VIEWPORT,
                user_agent=USER_AGENT,
                locale='ru-RU'
            )
            page = ctx.new_page()
            stealth_sync(page)

            print(
                "\n▶ Откройте сайт, авторизуйтесь, "
                "нажмите Enter в консоли…"
            )
            input()
            save_cookies(ctx)
            browser.close()
        logger.info("Куки сохранены.")
        return

    # Сбор задач в зависимости от режима
    if mode == 'link':
        target = input('Вставь ссылку: ').strip()
        tasks, limit = [(target, None)], max_links

    elif mode == 'csv':
        cars = load_models()
        trucks = load_models_truck()
        models = cars + trucks
        logger.info("Загружено %d авто-моделей и %d грузовых", len(cars), len(trucks))
        logger.info("\n%s", format_models_table(models))
        tasks = (
            generate_search_urls_cars(site)
            + generate_search_urls_truck(site)
        )
        if not tasks:
            logger.error("Нет ни одной модели для режима")
            return
        limit = n_links

    elif mode == 'llm':
        print('agent dont ready :(')
        return
    # Запуск браузера и сбор
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=40)
        ctx = browser.new_context(
            viewport=VIEWPORT,
            user_agent=USER_AGENT,
            locale='ru-RU',
            timezone_id='Europe/Warsaw'
        )
        load_cookies(ctx)
        page = ctx.new_page()
        for url, name in tasks:
            if stop_event.is_set():
                logger.warning("Прервано user")
                break
            process_target(page, url, name, known_links, limit, processed_counts)
        save_cookies(ctx)
        browser.close()

    # Обновление CSV моделей
    update_models_file(processed_counts)
    update_truck_models_file(processed_counts)

    total_time = time.time() - overall_start
    log_section("AGENT END")
    logger.info(
        "Models processed: %d, total time: %.2f s",
        len(processed_counts), total_time
    )

if __name__ == '__main__':
    main()
