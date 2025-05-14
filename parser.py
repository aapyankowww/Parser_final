import link_extr
import argparse
import processor
import yolo
import asyncio
def parse_args():
    parser = argparse.ArgumentParser(
        description="Сбор ссылок с Avito и Auto.ru"
    )
    parser.add_argument(
        '-m', '--mode',
        choices=['link', 'csv', 'llm', 'debug'],
        required=True,
    )
    parser.add_argument(
        '--n-links',
        type=int,
        default=10,
        help='Сколько ссылок собирать на каждую модель (N_LINKS)'
    )
    parser.add_argument(
        '--max-links',
        type=int,
        default=100,
        help='Лимит ссылок для работы по ссылке'
    )
    parser.add_argument(
        '--queue',
        choices=['extr', 'proc', 'yolo', 'full'],
        required=True,
    )
    return parser.parse_args()

def main():
    args = parse_args()
    mode = args.mode
    n_links = args.n_links
    max_links = args.max_links
    queue = args.queue
    if queue == 'extr':
        link_extr.main(mode=mode, n_links=n_links, max_links=max_links)
    elif queue == 'proc':
        asyncio.run(processor.main())
    elif queue == 'yolo':
        yolo.main()
    elif queue == 'full':
        link_extr.main(mode=mode, n_links=n_links, max_links=max_links)
        processor.main()
    else:
        print('Wrong mode')
        return
if __name__ == '__main__':
    main()