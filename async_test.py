import asyncio
import signal
import os

class value:
    def __init__(self):
        self.val = 1

async def count2(val_in: value):
    print("Three", val_in.val)
    val_in.val += 1
    await asyncio.sleep(1)
    print("Four", val_in.val)

async def count(val: value):
    print("One")
    #await asyncio.sleep(1)
    #await count2(val_in=val)
    asyncio.ensure_future(count2(val_in=val))

    #count2(val_in=val)
    print("Two")


async def count_by_one():
    count = 0
    while count < 5:
        print("By one: ", count)
        count += 1
        await asyncio.sleep(1)


async def count_by_two():
    count = 0
    while count < 10:
        print("By two: ", count)
        count += 2
        await asyncio.sleep(1.2)


# async def main():
#     val = value()
#     await asyncio.gather(count(val), count(val), count(val))
#     await asyncio.gather(count(val))
#     await asyncio.sleep(10)
#     await asyncio.sleep(0)


async def main():
    # tasks = []
    # loop = asyncio.get_event_loop()
    # # running = True

    # # def shutdown():
    # #     # Call shutdown code

    # #     # Indicuate shutdown
    # #     nonlocal running
    # #     running = False

    # # for sig in (signal.SIGINT, signal.SIGTERM):
    # #     loop.add_signal_handler(sig, shutdown)

    # # tasks.append(loop.create_task(count_by_one()))
    # # tasks.append(loop.create_task(count_by_two()))

    # # while running:
    # #     await asyncio.sleep(1)

    # tasks.append(asyncio.create_task(count_by_one()))
    # tasks.append(asyncio.create_task(count_by_two()))


    # # try:
    # #     # while True:
    # #     #     tasks = [t for t in tasks if not t.done()]
    # #     #     if len(tasks) == 0:
    # #     #         return
    # #     #     await tasks[0]

    # #     # loop = asyncio.get_event_loop()
    # #     loop.run_forever()

    # #     #await asyncio.gather(count_by_one(), count_by_two())
    # # except KeyboardInterrupt:
    # #     print("Interrupt received")


    # try:
    #     loop.run_until_complete(tasks)
    # except KeyboardInterrupt as e:


    tasks = []
    tasks.append(asyncio.create_task(count_by_one()))
    tasks.append(asyncio.create_task(count_by_two()))

    all_tasks = asyncio.all_tasks()
    all_tasks.remove(asyncio.current_task())
    await asyncio.wait(all_tasks)

    # try:
    #     while True:
    #         tasks = [t for t in tasks if not t.done()]
    #         if len(tasks) == 0:
    #             return
    #         await tasks[0]

    #     # loop = asyncio.get_event_loop()
    #     # loop.run_forever()

    #     #await asyncio.gather(count_by_one(), count_by_two())
    # except KeyboardInterrupt:
    #     print("Interrupt received")


if __name__ == "__main__":
    asyncio.run(main())
    #main()