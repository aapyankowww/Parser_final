import csv
import json
import logging
import os
import random
import re
import shutil
import tempfile
import uuid
from typing import Tuple
import asyncio
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from langchain_ollama import ChatOllama
from pydantic import BaseModel

logger = logging.getLogger()
AGENTS_FILE = 'user_agents.csv'
ROOT_DIR = 'photos'
LINKS_CSV = 'links.csv'
TIMEOUT = 100
BODY_CANON = {
    'sedan': ('седан', 'sedan'),
    'hatchback': ('хэтчбек', 'хетчбек', 'hatchback', 'хэтчбек 5-дверный', 'хетчбек 5 дв.',
                  'хэтчбек 3-дверный', 'хэтчбек 3 дв.', 'хетчбек 5 дв.', 'хетчбек 3 дв.',
                  'хэтчбек 5-дверный', 'хэтчбек 3-дверный'),
    'wagon': ('универсал', 'wagon'),
    'coupe': ('купе', 'coupe'),
    'convertible': ('кабриолет', 'кабрио', 'convertible'),
    'roadster': ('roadster', 'родстер'),
    'liftback': ('лифтбек', 'liftback'),
    'fastback': ('фастбек', 'fastback'),
    'microcar': ('микро', 'microcar'),
    'targa': ('targa', 'тарга'),
    'suv': ('джип', 'внедорожник', 'кроссовер', 'crossover', 'suv',
            'внедорожник 5-дверный', 'внедорожник 3-дверный', 'внедорожник 5 дв.',
            'внедорожник 3 дв.'),
    'minivan': ('минивэн', 'minivan'),
    'pickup': ('пикап', 'pickup'),
    #'flatbed': ('бортовой', 'flatbed', 'бортовой грузовик', 'рефрижератор', 'refrigerator'),
    'chassis': ('шасси', 'chassis'),
    'box': ('фургон-бокс', 'box'),
    'dump': ('самосвал', 'dump'),
    'bus': ('bus', 'микроавтобус', 'автобус'),
    'motorcycle': ('мотоцикл', 'мото', 'motorcycle', 'motorbike', 'bike'),
    'truck': ('грузовик', 'грузовой', 'truck', 'lorry', 'тентованный грузовик',
              'бортовой', 'flatbed', 'бортовой грузовик', 'рефрижератор', 'refrigerator'),
    'van': ('фургон', 'van', 'промтоварный фургон', 'изотермический фургон',
            'грузовой фургон', 'furgon', 'izotermic furgon'),
    'atv': ('квадроцикл', 'atv', 'quad'),
    'tractor': ('трактор', 'tractor'),
    'trailer': ('прицеп', 'полуприцеп', 'trailer', 'semi')
}


def mk_lookup(src: dict[str, Tuple[str, ...]]):
    d: dict[str, str] = {}
    for canon, aliases in src.items():
        for a in aliases + (canon,):
            d[a.lower()] = canon
    return d


BODY_LU = mk_lookup(BODY_CANON)


def start_browser():
    with open(AGENTS_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        agents = [row[0] for row in reader]
    agent = random.choice(agents)
    VIEWPORT = {'width': 1366, 'height': 1023}
    CTX_CFG = BrowserContextConfig(
        cookies_file="cookies.json",
        wait_for_network_idle_page_load_time=6.0,
        maximum_wait_page_load_time=120.0,
        browser_window_size=VIEWPORT,
        viewport_expansion=700,
        highlight_elements=True,
        user_agent=agent
    )
    browser = Browser()
    context = BrowserContext(browser=browser, config=CTX_CFG)
    controller = Controller(output_model=vehicle_data)
    return browser, context, controller


async def rotate_session(old_browser, old_context):
    try:
        await old_context.close()
    except Exception:
        pass
    try:
        await old_browser.close()
    except Exception:
        pass
    return start_browser()


def mark_done(target_url: str):
    tmp = tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', newline='')
    with open(LINKS_CSV, encoding='utf-8') as src, tmp:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(tmp, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['link'] == target_url:
                row['status'] = 'done'
            writer.writerow(row)
    shutil.move(tmp.name, LINKS_CSV)


def mark_err(target_url: str):
    tmp = tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', newline='')
    with open(LINKS_CSV, encoding='utf-8') as src, tmp:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(tmp, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['link'] == target_url:
                row['status'] = 'err'
            writer.writerow(row)
    shutil.move(tmp.name, LINKS_CSV)


PROMPT = '''
1. Извлеки brand, model, тип кузова (body_type).
2. Done
'''


class vehicle_data(BaseModel):
    body_type: str
    brand: str
    model: str

async def photo_load(page, url, body_type: str, brand: str, model: str):
    logger.info('=====START PHOTO_LOADER=====')
    try:
        # всё тело функции выполняется не дольше TIMEOUT
        async with asyncio.timeout(TIMEOUT):
            response = await page.goto(url, timeout=0)
            if not response or (response.status != 200) or (url != page.url):
                mark_err(url)
                if response and response.status == 429:
                    await asyncio.sleep(2)
                raise RuntimeError(f'Неверный URL {url}: статус {response.status if response else "error"}')

            if 'avito' in url:
                gallery = page.locator('[data-marker="item-view/gallery"]')
                next_btn = page.locator('[data-marker="extended-gallery-frame/control-right"]')
                image = page.locator('[data-marker="extended-gallery/frame-img"]')
                container = page.locator('[data-marker="extended-image-preview/item"]')
                await page.add_style_tag(content='''
                    .gallery-block-contactBarContainer-jtPAg{
                        display: none !important;
                    }
                    ''')
            elif 'auto.ru' in url:
                gallery = page.locator('[class*="ImageGalleryDesktop"]').first
                next_btn = page.locator(
                    '[class*="ImageGalleryFullscreenVertical__nav ImageGalleryFullscreenVertical__nav_right"]')
                image = page.locator('[class*="ImageGalleryFullscreenVertical__items"]')
                container = page.locator('[class*="ImageGalleryFullscreenVertical__thumbContainer"]')
            else:
                raise ValueError('Неизвестный сайт')

            box = await gallery.bounding_box()
            await page.mouse.click(box['x'] + box['width'] / 2, box['y'] + box['height'] / 3)
            await asyncio.sleep(2)

            save_dir = os.path.join(ROOT_DIR, body_type, brand, model)
            os.makedirs(save_dir, exist_ok=True)

            n_image = await container.count()
            if n_image == 1:
                mark_done(url)
                raise ValueError('Недостаточно ФМ')
            for i in range(n_image):
                path = os.path.join(save_dir, f'{uuid.uuid4()}.png')
                await page.mouse.move(10, 10, steps=5)
                if 'avito' in url:
                    try:
                        await image.screenshot(path=path)
                    except Exception:
                        pass
                elif 'auto.ru' in url:
                    try:
                        await image.locator(f'[data-index="{i}"]').screenshot(path=path)
                    except Exception:
                        pass
                await asyncio.sleep(2)
                await next_btn.click()
                await asyncio.sleep(1.5)

            mark_done(url)
            logger.info(f'DOWNLOADED: {n_image} images')
            logger.info('=====FINISH PHOTO_LOADER=====')

    except asyncio.TimeoutError:
        mark_err(url)
        logger.error(f'photo_load для {url} превысил таймаут {TIMEOUT} секунд')
        return


async def main():
    browser, context, controller = start_browser()
    try:
        if not os.path.exists(LINKS_CSV):
            logger.error('links.csv missing')
            return

        with open(LINKS_CSV, encoding='utf-8') as f:
            rows = [r for r in csv.DictReader(f) if not r.get('status')]

        if not rows:
            logger.info('Нет новых ссылок для обработки')
            await browser.close()
            return

        for row in rows:
            url = row['link'].strip()
            llm = ChatOllama(model='qwen2.5-coder:7b', num_ctx=6000, temperature=0.2, keep_alive=0)

            try:
                async with asyncio.timeout(5*TIMEOUT):
                    page = await context.get_current_page()
                    await page.add_init_script('window.open = () => null;')
                    page.on("dialog", lambda dialog: asyncio.create_task(dialog.dismiss()))

                    await page.context.grant_permissions(['geolocation'])
                    with open('cookies.json', encoding='utf-8') as f:
                        cookies = json.load(f)
                    await page.context.add_cookies(cookies)
                    await page.context.set_geolocation({'latitude': 59.960986, 'longitude': 30.284703})

                    response = await page.goto(url, timeout=0)
                    if not response or response.status != 200:
                        if (response.status == 429) or (response.status == 302):
                            browser, context, controller = await rotate_session(browser, context)
                            continue
                        mark_err(url)
                        raise RuntimeError(
                            f'Неверный URL {url}: статус {response.status if response else "no response"}'
                        )

                    agent = Agent(
                        task=PROMPT,
                        llm=llm,
                        browser_context=context,
                        controller=controller,
                        enable_memory=False,
                        max_actions_per_step=5,
                        use_vision=True,
                        max_failures=1
                    )

                    history = await agent.run(max_steps=4)
                    result = json.loads(history.final_result())
                    body, brand, model = result['body_type'], result['brand'], result['model']

                    if not all([body, brand, model]):
                        raise ValueError(f'Пустые поля: {result}')

                    body_c = BODY_LU.get(body.lower(), 'other')
                    brand_c = re.sub(r'\s+', '_', brand.strip().lower())
                    model_c = re.sub(r'\s+', '_', model.strip().lower())
                    logger.info(
                        f"LLM step has been finished with URL={page.url}, "
                        f"body={body_c}, brand={brand_c}, model={model_c}"
                    )

                    await photo_load(page, url, body_c, brand_c, model_c)

            except Exception as e:
                logger.error(f'Ошибка при обработке {url}:\n {e}')
            finally:
                try:
                    await page.close()
                except Exception:
                    pass
                await asyncio.sleep(1)
    finally:
        try:
            await context.close()
        except Exception:
            pass
        try:
            await browser.close()
        except Exception:
            pass


if __name__ == '__main__':
    asyncio.run(main())
